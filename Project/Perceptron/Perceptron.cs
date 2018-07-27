using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Perceptron {
	float[] weights;
	float learningRate;

	// Constructor
	public Perceptron (int n, float lr) {
		weights = new float[n];

		learningRate = lr;

		// Initialize the weights randomly
		for (int i = 0; i < weights.Length; ++i) {
			weights [i] = Random.Range (-1.0f, 1.0f);
		}
	}

	public int guess(float[] inputs){
		float sum = 0;

		for (int i = 0; i < weights.Length; ++i) {
			sum += inputs [i] * weights [i];
		}

		int output = sign(sum);
		return output;
	}

	public void train(float[] inputs, int target){
		int doGuess = guess (inputs);
		int error = target - doGuess;

		// Tweak the weights based on the error, if we guessed correctly, there will be nor error so no tweak
		for (int i = 0; i < weights.Length; ++i) {
			weights [i] += error * inputs [i] * learningRate;
		}
	}

	public float guessY(float x) {
		float w0 = weights [0];
		float w1 = weights [1];
		float w2 = weights [2];

		return -(w2 / w1) - (w0 / w1) * x;
	}

	// The activation function
	int sign(float n){
		if (n >= 0) {
			return 1;
		} else {
			return -1;
		}
	}
}