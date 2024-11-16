// Copyright 2024 The Vino Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"io"
	"math"
	"math/rand"
	"strconv"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Index    int
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
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for i := range item[:4] {
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

func main() {
	rng := rand.New(rand.NewSource(1))
	data := Load()
	u := make([]float64, 4)
	for _, item := range data {
		for i, v := range item.Measures {
			u[i] += v
		}
	}
	n := float64(len(data))
	for i, v := range u {
		u[i] = v / n
	}
	s := make([]float64, 4)
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
		measures := make([]float64, 4)
		for j := range measures {
			measures[j] = s[j]*rng.NormFloat64() + u[j]
			data = append(data, Fisher{
				Measures: measures,
				Label:    "Iris-fake",
				Index:    length + i,
			})
		}
	}
}
