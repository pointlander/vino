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
	// Width is the width of the model
	Width = 4
	// Factor is the gaussian factor
	Factor = .1
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Index    int
	S        float64
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
					Label:    item[4],
					Index:    i,
				}
				for i := range item[:Width] {
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
	u := make([]float64, Width)
	for _, item := range data {
		for i, v := range item.Measures {
			u[i] += v
		}
	}
	n := float64(len(data))
	for i, v := range u {
		u[i] = v / n
	}
	s := make([]float64, Width)
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
		measures := make([]float64, Width)
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
					x := make([]float64, Width)
					for k := range x {
						x[k] = rng.NormFloat64() * Factor
					}
					y := make([]float64, Width)
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
				x := make([]float64, Width)
				y := make([]float64, Width)
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
				x := make([]float64, Width)
				y := make([]float64, Width)
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
				x := make([]float64, Width)
				y := make([]float64, Width)
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
	}
	networks := make([]Network, 3)
	for n := range networks {
		set := tf64.NewSet()
		set.Add("w1", 4, 3)
		set.Add("b1", 3)
		set.Add("w2", 3, 4)
		set.Add("b2", 4)

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
		others.Add("input", 4)
		others.Add("output", 4)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
		l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
		loss := tf64.Quadratic(l2, others.Get("output"))
		networks[n].Set = set
		networks[n].Others = others
		networks[n].L1 = l1
		networks[n].L2 = l2
		networks[n].Loss = loss
	}

	points := make(plotter.XYs, 0, 8)
	for i := 0; i < 33*len(data); i++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		acc, network := make([]float64, len(networks)), 0
		for n := range networks {
			networks[n].Others.Zero()
			index := rng.Intn(len(data))
			input := networks[n].Others.ByName["input"].X
			for j := range input {
				input[j] = data[index].Measures[j]
			}
			output := networks[n].Others.ByName["output"].X
			for j := range output {
				output[j] = data[index].Measures[j]
			}
			networks[n].Loss(func(a *tf64.V) bool {
				acc[n] = a.X[0]
				return true
			})
		}
		max, sum := 0.0, 0.0
		for _, v := range acc {
			if v > max {
				max = v
			}
		}
		max *= 2.0
		for _, v := range acc {
			sum += (max - v)
		}
		type A struct {
			A float64
			I int
		}
		a := make([]A, len(acc))
		for i, v := range acc {
			a[i].A = (max - v) / sum
			a[i].I = i
		}
		sort.Slice(a, func(i, j int) bool {
			return a[i].A < a[j].A
		})
		sum = 0
		s := rng.Float64()
		for _, v := range a {
			sum += v.A
			if s < sum {
				network = v.I
				break
			}
		}

		networks[network].Others.Zero()
		index := rng.Intn(len(data))
		input := networks[network].Others.ByName["input"].X
		for j := range input {
			input[j] = data[index].Measures[j]
		}
		output := networks[network].Others.ByName["output"].X
		for j := range output {
			output[j] = data[index].Measures[j]
		}

		networks[network].Set.Zero()
		cost := tf64.Gradient(networks[network].Loss).X[0]

		norm := 0.0
		for _, p := range networks[network].Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		if norm > 1 {
			scaling := 1 / norm
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
		} else {
			for _, w := range networks[network].Set.Weights {
				for l, d := range w.D {
					g := d
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
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
	}

	for index := range data {
		min, network := math.MaxFloat64, 0
		for n := range networks {
			networks[n].Others.Zero()
			input := networks[n].Others.ByName["input"].X
			for j := range input {
				input[j] = data[index].Measures[j]
			}
			output := networks[n].Others.ByName["output"].X
			for j := range output {
				output[j] = data[index].Measures[j]
			}
			networks[n].Loss(func(a *tf64.V) bool {
				acc := a.X[0]
				if acc < min {
					min, network = acc, n
				}
				return true
			})
		}
		fmt.Println(network, data[index].Label)
	}

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
