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

	"github.com/pointlander/vino/kmeans"

	"github.com/alixaxel/pagerank"
)

const (
	// Width is the width of the model
	Width = 4
	// Factor is the gaussian factor
	Factor = 0.1
)

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
}
