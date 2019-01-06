package main

import (
	"fmt"
	"math"
)

func NDCG( targetSet map[uint64]bool, rankedSet []uint64, K uint64) float64 {
	/*
		Function:
		Calculates NDCG scores for binary relevance scenario.
		Takes in a target set i.e. elements in this set have relevance 1.0 and rest have 0.0. Also takes input
		a rankedSet where the top ranked items retrieved are present and K which decides NDCG@K.
		Arg:
		targetSet - map from id to bool
		rankedSet - array of ids in order
		K - index till which NDCG is calculated
	*/
	 idcg := 0.0
	 var i uint64
	 for i = 0; i < uint64(len(targetSet)); i++ {
		if i < K {
			idcg += 1.0 / math.Log2(float64(i)+2.0)
		}
	}

	dcg := 0.0
	for i, itemId := range rankedSet {
		if _, exist := targetSet[itemId]; exist {
			dcg += 1.0 / math.Log2(float64(i)+2.0)
		}
	}
	return dcg/idcg
}

// test
// func main() {
// 	rankset := []uint64{10, 20, 30, 40}
// 	targetset := map[uint64]bool{ 10: true, 50: true, 70: true, 30:true }
// 	var K uint64 = 4
// 	score := NDCG(targetset, rankset, K)
// 	fmt.Println(" score ", score)
// }
