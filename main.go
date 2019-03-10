package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// sigmoid Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// derivSigmoid Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
func derivSigmoid(x float64) float64 {
	fx := sigmoid(x)
	return fx * (1 - fx)
}

// mse Mean Squared Error
func mse(inputs []Input, predictions []float64) float64 {
	len := len(inputs)

	var total float64
	for i := 0; i < len; i++ {
		total += math.Pow(inputs[i].Expected-predictions[i], 2)
	}

	return total / float64(len)
}

// ApplyInputToNeuron returns activation and result
func ApplyInputToNeuron(i Input, n Neuron) (float64, float64) {
	result := n.Weight1*i.One + n.Weight2*i.Two + n.Bias
	activation := sigmoid(result)
	return activation, result
}

// Predict generates the prediction
func Predict(n1, n2, n3 Neuron, i Input) float64 {
	_, n1Activation := ApplyInputToNeuron(i, n1)
	_, n2Activation := ApplyInputToNeuron(i, n2)
	_, n3Activation := ApplyInputToNeuron(Input{n1Activation, n2Activation, 0.0}, n3)

	return n3Activation
}

// Input represents a input
type Input struct {
	One      float64
	Two      float64
	Expected float64
}

// Neuron represents a single neuron
type Neuron struct {
	Weight1 float64
	Weight2 float64
	Bias    float64
}

// GenNeuron returns a new randomized neuron
func GenNeuron() Neuron {
	return Neuron{rand.NormFloat64(), rand.NormFloat64(), rand.NormFloat64()}
}

// GenTrainingData generates the training Data
func GenTrainingData() []Input {

	var inputs []Input

	csvFile, _ := os.Open("inputs.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))

	tWeight := 0.0
	tHeight := 0.0

	for {
		line, error := reader.Read()
		if error == io.EOF {
			break
		} else if error != nil {
			log.Fatal(error)
		}
		expected := 1.0
		if line[0] == "Male" {
			expected = 0.0
		}
		weight, _ := strconv.ParseFloat(line[2], 64)
		height, _ := strconv.ParseFloat(line[1], 64)

		tWeight += weight
		tHeight += height

		inputs = append(inputs, Input{
			One:      weight,
			Two:      height,
			Expected: expected,
		})
	}

	avgHeight := tHeight / float64(len(inputs))
	avgWeight := tWeight / float64(len(inputs))

	for i := range inputs {
		inputs[i].One -= avgHeight
		inputs[i].Two -= avgWeight
	}

	return inputs
}

func main() {

	n1 := GenNeuron()
	n2 := GenNeuron()
	n3 := GenNeuron()

	inputs := GenTrainingData()

	learnRate := 0.0001
	epochs := 4000

	for i := 1; i <= epochs; i++ {
		for _, input := range inputs {

			n1Result, n1Activation := ApplyInputToNeuron(input, n1)
			n2Result, n2Activation := ApplyInputToNeuron(input, n2)
			n3Result, n3Activation := ApplyInputToNeuron(Input{n1Activation, n2Activation, 0.0}, n3)

			prediction := n3Activation

			pLpPrediction := -2 * (input.Expected - prediction)

			// Neuron 3 (output)
			pPredictionpW5 := n1Activation * derivSigmoid(n3Result)
			pPredictionpW6 := n2Activation * derivSigmoid(n3Result)
			pPredictionpB3 := derivSigmoid(n3Result)

			pPredictionpN1Activation := n3.Weight1 * derivSigmoid(n3Result)
			pPredictionpN2Activation := n3.Weight2 * derivSigmoid(n3Result)

			// Neuron 1
			pN1ActivationpN1W1 := input.One * derivSigmoid(n1Result)
			pN1ActivationpN1W2 := input.Two * derivSigmoid(n1Result)
			pN1ActivationpN1B1 := derivSigmoid(n1Result)

			// Neuron 2
			pN2ActivationpW3 := input.One * derivSigmoid(n2Result)
			pN2ActivationpW4 := input.Two * derivSigmoid(n2Result)
			pN2ActivationpB2 := derivSigmoid(n2Result)

			n1.Weight1 -= learnRate * pLpPrediction * pPredictionpN1Activation * pN1ActivationpN1W1
			n1.Weight2 -= learnRate * pLpPrediction * pPredictionpN1Activation * pN1ActivationpN1W2
			n1.Bias -= learnRate * pLpPrediction * pPredictionpN1Activation * pN1ActivationpN1B1

			n2.Weight1 -= learnRate * pLpPrediction * pPredictionpN2Activation * pN2ActivationpW3
			n2.Weight2 -= learnRate * pLpPrediction * pPredictionpN2Activation * pN2ActivationpW4
			n2.Bias -= learnRate * pLpPrediction * pPredictionpN2Activation * pN2ActivationpB2

			n3.Weight1 -= learnRate * pLpPrediction * pPredictionpW5
			n3.Weight2 -= learnRate * pLpPrediction * pPredictionpW6
			n3.Bias -= learnRate * pLpPrediction * pPredictionpB3
		}

		if i%1000 == 0 {
			var dPredictions []float64
			for _, input := range inputs {
				dPredictions = append(dPredictions, Predict(n1, n2, n3, input))
			}
			loss := mse(inputs, dPredictions)
			fmt.Printf("Epoch %d loss %.4f\n", i, loss)
		}
	}

	a1 := Predict(n1, n2, n3, Input{136.6870, 69.6850, +1.0})
	a2 := Predict(n1, n2, n3, Input{244.7130, 72.0472, +0.0})

	fmt.Printf("Fabi expected is %.4f\n", a1)
	fmt.Printf("Guilherme expected is %.4f\n", a2)

}
