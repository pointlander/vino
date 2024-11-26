// Copyright 2024 The Vino Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/vino/kmeans"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/alixaxel/pagerank"
)

const (
	// Inputs is the width of the inputs
	Inputs = 4
	// Width is the width of the model
	Width = 8
	// Embedding is the embedding size
	Embedding = Width
	// Factor is the gaussian factor
	Factor = 0.01
	// Batch is the batch size
	Batch = 16
	// Networks is the number of networks
	Networks = 3
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// Dot computes the dot product
func dot(x, y []float64) (z float64) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float32 matrices
func (m Matrix) Add(n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

func (m Matrix) Softmax() Matrix {
	output := NewMatrix(m.Cols, m.Rows)
	max := 0.0
	for _, v := range m.Data {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	values := make([]float64, len(m.Data))
	for j, value := range m.Data {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for _, value := range values {
		output.Data = append(output.Data, value/sum)
	}
	return output
}

// T tramsposes a matrix
func (m Matrix) T() Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float64, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float64, V.Cols), make([]float64, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) []float64 {
	entropies, values, results := make([]float64, V.Cols), make([]float64, K.Rows), make([]float64, 0, K.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot(values, V)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, -entropy)
	}
	return results
}

// MakeRandomTransform makes a random transform
func MakeRandomTransform(rng *rand.Rand, cols, rows int, stddev float64) Matrix {
	transform := NewMatrix(cols, rows)
	for i := 0; i < rows; i++ {
		row := NewMatrix(cols, 1)
		for j := 0; j < cols; j++ {
			row.Data = append(row.Data, math.Abs(rng.NormFloat64())*stddev)
		}
		row = row.Softmax()
		for _, v := range row.Data {
			transform.Data = append(transform.Data, v)
		}
	}
	return transform
}

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Index    int
	S        float64
	Votes    [3]int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
	"Iris-fake":       3,
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, Width),
					Label:    item[Inputs],
					Index:    i,
				}
				for i := range item[:Inputs] {
					f, err := strconv.ParseFloat(item[i], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[i] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// Dot computes the dot product
func Dot(a, b, x, y []float64) (z float64) {
	for i := range a {
		z += (a[i] + x[i]) * (b[i] + y[i])
	}
	return z
}

// AddRandom adds random training examples
func AddRandom(data []Fisher, rng *rand.Rand) []Fisher {
	u := make([]float64, Inputs)
	for _, item := range data {
		for i, v := range item.Measures {
			u[i] += v
		}
	}
	n := float64(len(data))
	for i, v := range u {
		u[i] = v / n
	}
	s := make([]float64, Inputs)
	for _, item := range data {
		for i, v := range item.Measures {
			d := v - u[i]
			s[i] += d * d
		}
	}
	for i, v := range s {
		s[i] = math.Sqrt(v / n)
	}
	length := len(data)
	for i := 0; i < 50; i++ {
		measures := make([]float64, Inputs)
		for j := range measures {
			measures[j] = s[j]*rng.NormFloat64() + u[j]
		}
		data = append(data, Fisher{
			Measures: measures,
			Label:    "Iris-fake",
			Index:    length + i,
		})
	}

	return data
}

// Random is random mode
func Random() {
	rng := rand.New(rand.NewSource(1))
	data := Load()
	data = AddRandom(data, rng)
	length := len(data)
	{
		ranks := make([][]float64, 0, 32)
		for e := 0; e < 32; e++ {
			graph := pagerank.NewGraph()
			for i, a := range data {
				for j, b := range data {
					x := make([]float64, Inputs)
					for k := range x {
						x[k] = rng.NormFloat64() * Factor
					}
					y := make([]float64, Inputs)
					for k := range y {
						y[k] = rng.NormFloat64() * Factor
					}
					aa := 0.0
					for k, v := range a.Measures {
						v += x[k]
						aa += v * v
					}
					bb := 0.0
					for k, v := range b.Measures {
						v += y[k]
						bb += v * v
					}
					weight := Dot(a.Measures, b.Measures, x, y)
					graph.Link(uint32(i), uint32(j), math.Abs(weight)/(aa*bb))
				}
			}
			r := make([]float64, length)
			graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
				r[node] = rank
			})
			ranks = append(ranks, r)
			fmt.Printf(".")
		}
		fmt.Println()
		u := make([]float64, length)
		for _, r := range ranks {
			for i, v := range r {
				u[i] += v
			}
		}
		for i, v := range u {
			u[i] = v / float64(length)
		}
		s := make([]float64, length)
		for _, r := range ranks {
			for i, v := range r {
				d := v - u[i]
				s[i] += d * d
			}
		}
		for i, v := range s {
			s[i] = math.Sqrt(v / float64(length))
		}
		for i := range data {
			data[i].S = s[i]
		}
		sort.Slice(data, func(i, j int) bool {
			return data[i].S > data[j].S
		})
		for _, item := range data {
			fmt.Println(item.S, item.Label)
		}
	}
}

// Iterative is the iterative mode
func Iterative() {
	rng := rand.New(rand.NewSource(1))
	data := Load()
	data = AddRandom(data, rng)
	output := []Fisher{}
	for e := 0; e < 199; e++ {
		graph := pagerank.NewGraph()
		for i, a := range data {
			for j, b := range data {
				x := make([]float64, Inputs)
				y := make([]float64, Inputs)
				aa := 0.0
				for k, v := range a.Measures {
					v += x[k]
					aa += v * v
				}
				bb := 0.0
				for k, v := range b.Measures {
					v += y[k]
					bb += v * v
				}
				weight := Dot(a.Measures, b.Measures, x, y)
				graph.Link(uint32(i), uint32(j), math.Abs(weight)/(aa*bb))
			}
		}
		r := make([]float64, len(data))
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			r[node] = rank
		})
		for i := range data {
			data[i].S = r[i]
		}
		sort.Slice(data, func(i, j int) bool {
			return data[i].S > data[j].S
		})
		output = append(output, data[len(data)-1])
		data = data[:len(data)-1]
	}
	for _, item := range output {
		fmt.Println(item.Label, item.S)
	}
}

// Differential is the differential mode
func Differential() {
	rng := rand.New(rand.NewSource(1))
	data := Load()
	data = AddRandom(data, rng)
	output := make([][]float64, 0, 200)
	for e := 0; e < 200; e++ {
		dgraph := pagerank.NewGraph()
		for i, a := range data {
			if e == i {
				continue
			}
			for j, b := range data {
				if e == j {
					continue
				}
				x := make([]float64, Inputs)
				y := make([]float64, Inputs)
				aa := 0.0
				for k, v := range a.Measures {
					v += x[k]
					aa += v * v
				}
				bb := 0.0
				for k, v := range b.Measures {
					v += y[k]
					bb += v * v
				}
				weight := Dot(a.Measures, b.Measures, x, y)
				dgraph.Link(uint32(i), uint32(j), math.Abs(weight)/(aa*bb))
			}
		}
		dr := make([]float64, len(data))
		dgraph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			dr[node] = rank
		})

		graph := pagerank.NewGraph()
		for i, a := range data {
			for j, b := range data {
				x := make([]float64, Inputs)
				y := make([]float64, Inputs)
				aa := 0.0
				for k, v := range a.Measures {
					v += x[k]
					aa += v * v
				}
				bb := 0.0
				for k, v := range b.Measures {
					v += y[k]
					bb += v * v
				}
				weight := Dot(a.Measures, b.Measures, x, y)
				graph.Link(uint32(i), uint32(j), math.Abs(weight)/(aa*bb))
			}
		}
		r := make([]float64, len(data))
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			r[node] = rank
		})
		for i := range r {
			r[i] = dr[i] - r[i]
		}
		cp := make([]Fisher, len(data))
		copy(cp, data)
		for i := range cp {
			cp[i].S = r[i]
		}
		sort.Slice(cp, func(i, j int) bool {
			return cp[i].S > cp[j].S
		})
		for i := range r {
			r[i] = float64(cp[i].Index)
		}
		output = append(output, r)
	}
	meta := make([][]float64, len(output))
	for i := range meta {
		meta[i] = make([]float64, len(output))
	}
	for i := 0; i < 256; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), output, 4, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(output); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 4, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, item := range data {
		fmt.Println(clusters[i], item.Label)
	}
}

var (
	// FlagRandom is the random mode
	FlagRandom = flag.Bool("random", false, "random mode")
	// FlagIterative is the iterative mode
	FlagIterative = flag.Bool("iterative", false, "iterative mode")
	// FlagDifferential is the differential mode
	FlagDifferential = flag.Bool("differential", false, "differential mode")
)

func main() {
	flag.Parse()

	if *FlagRandom {
		Random()
		return
	} else if *FlagIterative {
		Iterative()
		return
	} else if *FlagDifferential {
		Differential()
		return
	}

	data := Load()
	rng := rand.New(rand.NewSource(1))

	type Network struct {
		Set    tf64.Set
		Others tf64.Set
		L1     tf64.Meta
		L2     tf64.Meta
		Loss   tf64.Meta
		I      int
		V      tf64.Meta
		VV     tf64.Meta
		E      tf64.Meta
	}
	networks := make([]Network, Networks)
	for n := range networks {
		set := tf64.NewSet()
		set.Add("w1", Embedding, Width/2)
		set.Add("b1", Width/2)
		set.Add("w2", Width/2, Inputs)
		set.Add("b2", Inputs)

		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
		}

		others := tf64.NewSet()
		others.Add("input", Embedding, Batch)
		others.Add("output", Inputs, Batch)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
		l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
		loss := tf64.Quadratic(l2, others.Get("output"))
		v := tf64.Variance(loss)
		e := tf64.Entropy(tf64.Softmax(loss))
		diff := tf64.Sub(l2, others.Get("output"))
		vv := tf64.Avg(tf64.Variance(tf64.T(tf64.Hadamard(diff, diff))))
		networks[n].Set = set
		networks[n].Others = others
		networks[n].L1 = l1
		networks[n].L2 = l2
		networks[n].Loss = tf64.Avg(loss)
		networks[n].V = v
		networks[n].VV = vv
		networks[n].E = e
	}

	pow := func(x float64, i int) float64 {
		y := math.Pow(x, float64(i+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}

	points := make(plotter.XYs, 0, 8)
	buffer := NewMatrix(Inputs, 8, make([]float64, 4*8)...)
	b := 0
	for i := 0; i < 4*33*len(data); i++ {
		index := rng.Intn(len(data))
		network, min := 0, math.MaxFloat64
		copy(buffer.Data[b*Inputs:(b+1)*Inputs], data[index].Measures)
		measures := SelfAttention(buffer, buffer, buffer)
		dat := append(measures.Data, data[index].Measures...)
		for s := 0; s < Batch; s++ {
			transform := MakeRandomTransform(rng, Width, Embedding, Factor)
			//offset := MakeRandomTransform(rng, Width, 1, Scale)
			in := NewMatrix(Width, 1, dat[b*8:(b+1)*8]...)
			in = transform.MulT(in) //.Add(offset).Softmax()
			for n := range networks {
				copy(networks[n].Others.ByName["input"].X[s*Embedding:(s+1)*Embedding], in.Data)
				copy(networks[n].Others.ByName["output"].X[s*Inputs:(s+1)*Inputs], data[index].Measures)
			}
		}
		b = (b + 1) % 8
		for n := range networks {
			networks[n].Others.Zero()
			networks[n].V(func(a *tf64.V) bool {
				if a.X[0] < min {
					min, network = a.X[0], n
				}
				return true
			})
		}

		networks[network].Others.Zero()

		networks[network].Set.Zero()
		cost := tf64.Gradient(networks[network].Loss).X[0]

		norm := 0.0
		for _, p := range networks[network].Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1, networks[network].I), pow(B2, networks[network].I)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range networks[network].Set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		networks[network].I++
	}

	histogram := [3][Networks]float64{}
	for j := range buffer.Data {
		buffer.Data[j] = 0
	}
	b = 0
	for shot := 0; shot < 33; shot++ {
		for range data {
			index := rng.Intn(len(data))
			network, min := 0, math.MaxFloat64
			copy(buffer.Data[b*Inputs:(b+1)*Inputs], data[index].Measures)
			measures := SelfAttention(buffer, buffer, buffer)
			dat := append(measures.Data, data[index].Measures...)
			for s := 0; s < Batch; s++ {
				transform := MakeRandomTransform(rng, Width, Embedding, Factor)
				//offset := MakeRandomTransform(rng, Width, 1, Scale)
				in := NewMatrix(Width, 1, dat[b*8:(b+1)*8]...)
				in = transform.MulT(in) //.Add(offset).Softmax()
				for n := range networks {
					copy(networks[n].Others.ByName["input"].X[s*Embedding:(s+1)*Embedding], in.Data)
					copy(networks[n].Others.ByName["output"].X[s*Inputs:(s+1)*Inputs], data[index].Measures)
				}
			}
			b = (b + 1) % 8
			for n := range networks {
				networks[n].Others.Zero()
				networks[n].V(func(a *tf64.V) bool {
					if a.X[0] < min {
						min, network = a.X[0], n
					}
					return true
				})
			}
			data[index].Votes[network]++
		}
	}
	for _, item := range data {
		max, index := 0, 0
		for i, v := range item.Votes {
			if v > max {
				max, index = v, i
			}
		}
		histogram[Labels[item.Label]][index]++
	}
	fmt.Println(histogram)

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

}
