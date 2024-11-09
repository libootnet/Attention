# Attention

### This repo is a simplified implementation of attention theory in the Go.
<br />
The following implementation is a Go implementation of

```
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(dK)) * V
```

Implemented in Go

```go
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
```

This function performs the forward computation for a Linear (fully connected) layer by applying the weight matrix W and bias vector B to the input matrix X.

```go
func (l *Linear) Forward(x Matrix) Matrix {
	y := MatMul(x, l.W)
	for i := range y {
		for j := range y[i] {
			y[i][j] += l.B[j]
		}
	}
	return y
}
```
