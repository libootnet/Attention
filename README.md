# Attention

### This repo is a simplified implementation of attention theory in the Go.
<br />
The following implementation is a Go implementation of

```
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(dK)) * V
```

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
