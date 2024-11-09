package main

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	dModel = 64
	dK     = 64
	dV     = 64
)

type Matrix [][]float64

func NewMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

type Linear struct {
	W Matrix
	B []float64
}

func NewLinear(inputDim, outputDim int) *Linear {
	linear := &Linear{
		W: NewMatrix(outputDim, inputDim),
		B: make([]float64, outputDim),
	}
	for i := range linear.W {
		for j := range linear.W[i] {
			linear.W[i][j] = rand.NormFloat64() * 0.01
		}
		linear.B[i] = rand.NormFloat64() * 0.01
	}
	return linear
}

func MatMul(a, b Matrix) Matrix {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])
	if colsA != rowsB {
		panic("size not match")
	}
	result := NewMatrix(rowsA, colsB)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

func (l *Linear) Forward(x Matrix) Matrix {
	y := MatMul(x, l.W)
	for i := range y {
		for j := range y[i] {
			y[i][j] += l.B[j]
		}
	}
	return y
}

// Attention(Q, K, V) = softmax((Q * K^T) / sqrt(dK)) * V
func ScaledDotProductAttention(query, key, value Matrix) Matrix {
	scaleFactor := 1.0 / math.Sqrt(float64(dK))
	for i := range query {
		for j := range query[i] {
			query[i][j] *= scaleFactor
		}
	}

	// softmax
	scores := MatMul(query, Transpose(key))

	for i := range scores {
		rowMax := math.Inf(-1)
		for _, v := range scores[i] {
			if v > rowMax {
				rowMax = v
			}
		}
		sumExp := 0.0
		for j := range scores[i] {
			scores[i][j] = math.Exp(scores[i][j] - rowMax)
			sumExp += scores[i][j]
		}
		for j := range scores[i] {
			scores[i][j] /= sumExp
		}
	}

	output := MatMul(scores, value)
	return output
}

func Transpose(m Matrix) Matrix {
	rows, cols := len(m), len(m[0])
	result := NewMatrix(cols, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = m[i][j]
		}
	}
	return result
}

func RandomMatrix(rows, cols int) Matrix {
	matrix := NewMatrix(rows, cols)
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = rand.NormFloat64()
		}
	}
	return matrix
}

func main() {
	// query := NewMatrix(1, dK)
	// key := NewMatrix(1, dK)
	// value := NewMatrix(1, dV)

	query := RandomMatrix(1, dK)
	key := RandomMatrix(1, dK)
	value := RandomMatrix(1, dV)

	attentionOutput := ScaledDotProductAttention(query, key, value)
	fmt.Println("Attention Output Value is ", attentionOutput)

	linear := NewLinear(dModel, dModel)
	linearOutput := linear.Forward(attentionOutput)
	fmt.Println("Linear Output Value is ", linearOutput)
}
