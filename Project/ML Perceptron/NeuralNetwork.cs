using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActivationFunction{
	// The current function being used
	public Matrix.mapFunc func;
	// The d function of the current function being used
	public Matrix.mapFunc dfunc;

	// Construct with a function and d function
	public ActivationFunction(Matrix.mapFunc func_, Matrix.mapFunc dfunc_){
		func = func_;
		dfunc = dfunc_;
	}
}

// Example will allow for a 3 layered Neural Network only
public class NeuralNetwork {
	public int input_nodes;
	public int hidden_nodes;
	public int output_nodes;

	public Matrix weights_ih;
	public Matrix weights_ho;

	public Matrix bias_h;
	public Matrix bias_o;


	// If batch processing, cumulative matrices
	public Matrix batchWeights_ih;
	public Matrix batchWeights_ho;

	public Matrix batchBias_h;
	public Matrix batchBias_o;


	public float learningRate;

	public static ActivationFunction sigmoid;
	public static ActivationFunction tanh;
	public static ActivationFunction relu;

	public ActivationFunction activeFunc;

	public float cumError = 0;

	private int posInBatch = 0;

	public Texture2D activationTexture;
	public Texture2D softMaxTexture;

	public NeuralNetwork (int numI, int numH, int numO){
		// Define a sigmoid activation and derrivative function
		sigmoid = new ActivationFunction (
			x => 1 / (1 + Mathf.Exp(-x)), 
			y => y * (1 - y)
		);
		// Define a tanh activation and derrivative function
		tanh = new ActivationFunction (
			x => (float)System.Math.Tanh(x), 
			y => 1 - (y * y)
		);

		// Define a ReLu activation and derrivative function
		relu = new ActivationFunction (
			x => (x <= 0) ? 0 : x, 
			y => (y <= 0) ? 0 : 1
		);
			
		input_nodes = numI;
		hidden_nodes = numH;
		output_nodes = numO;

		// 2 sets of weights between I -> H -> O
		weights_ih = new Matrix (hidden_nodes, input_nodes);
		weights_ho = new Matrix (output_nodes, hidden_nodes);
		weights_ih.randomize ();
		weights_ho.randomize ();

		// Bias value for every node
		bias_h = new Matrix (hidden_nodes, 1);
		bias_o = new Matrix (output_nodes, 1);
		bias_h.randomize ();
		bias_o.randomize ();

		batchWeights_ih = new Matrix (hidden_nodes, input_nodes);
		batchWeights_ho = new Matrix (output_nodes, hidden_nodes);

		batchBias_h = new Matrix (hidden_nodes, 1);
		batchBias_o = new Matrix (output_nodes, 1);

		setActivationFunction (sigmoid);
		setLearningRate (0.3f);
	}

	//Apply a function to every element of matrix
	public void setActivationFunction (ActivationFunction inFunc)
	{
		activeFunc = inFunc;
	}

	public void setLearningRate(float learning_rate = 0.1f){
		learningRate = learning_rate;
	}

	int softMax(float[] input_array){

		float[] z_exp = new float[input_array.Length];

		float sum_z_exp = 0;
			
		for(int i = 0; i < input_array.Length; ++i){

			z_exp[i] = Mathf.Exp(input_array[i]);

			sum_z_exp += z_exp [i];
		}

		float[] softmaxOut = new float[input_array.Length];

		for (int i = 0; i < input_array.Length; ++i) {
			softmaxOut [i] = z_exp [i] / sum_z_exp;
		}

		softMaxTexture = new Texture2D (1, output_nodes);

		Color[] cols = new Color[softmaxOut.Length];

		for (int i = 0; i < softmaxOut.Length; ++i) {
			cols [i] = new Color(softmaxOut [i], softmaxOut [i], softmaxOut [i]);
		}

		softMaxTexture.SetPixels (cols);
		softMaxTexture.filterMode = FilterMode.Point;
		softMaxTexture.Apply();

		int guess = 0;
		float highest = 0;

		float cumulative = 0;

		for (int j = 0; j < softmaxOut.Length; ++j) {
			cumulative += softmaxOut [j];
			if (softmaxOut [j] > highest) {
				highest = softmaxOut [j];
				guess = j;
			}
		}
			
		return guess;
	}

	public Matrix predictNOSoft(float[] input_array){
		//Generating the Hidden Outputs
		Matrix inputs = Matrix.fromArray (input_array);
		Matrix hiddens = Matrix.multiply (weights_ih, inputs);
		hiddens.add (bias_h);
		//Activation
		hiddens.map (activeFunc.func);

		Matrix outputs = Matrix.multiply(weights_ho, hiddens);
		outputs.add (bias_o);
		// Pass sigmoid as a delegate
		outputs.map(activeFunc.func);

		return outputs;
	}

	public int predict(float[] input_array){
		//Generating the Hidden Outputs
		Matrix inputs = Matrix.fromArray (input_array);
		Matrix hiddens = Matrix.multiply (weights_ih, inputs);
		hiddens.add (bias_h);
		//Activation
		hiddens.map (activeFunc.func);

		activationTexture = new Texture2D (1, hidden_nodes);

		Color[] cols = new Color[hidden_nodes];

		for(int i = 0; i < hidden_nodes; ++i){

			float val = hiddens.data [i, 0] < 0 ? 1 : hiddens.data [i, 0];
			float val2 = hiddens.data [i, 0] < 0 ? 0 : hiddens.data [i, 0];

			cols [i] = new Color (val, val2, val2);
		}

		activationTexture.SetPixels(cols);
		activationTexture.filterMode = FilterMode.Point;
		activationTexture.Apply ();

		Matrix outputs = Matrix.multiply(weights_ho, hiddens);
		outputs.add (bias_o);
		// Pass sigmoid as a delegate
		outputs.map(activeFunc.func);

		return softMax (outputs.toArray());
	}

	public void train(float[] input_array, float[] target_array){
		//Generating the Hidden Outputs
		Matrix inputs = Matrix.fromArray (input_array);
		Matrix hiddens = Matrix.multiply (weights_ih, inputs);
		hiddens.add (bias_h);
		//Activation
		hiddens.map (activeFunc.func);

		Matrix outputs = Matrix.multiply(weights_ho, hiddens);
		outputs.add (bias_o);

		// Pass sigmoid as a delegate
		outputs.map(activeFunc.func);

		Matrix targets = Matrix.fromArray (target_array);

		//----------
		// Calculate the output errors
		Matrix output_errors = Matrix.subtract(targets, outputs);


		// Calculate gradient
		Matrix gradients = Matrix.map(outputs, activeFunc.dfunc);
		gradients.multiply (output_errors);

		gradients.multiply (learningRate);

		// Calculate deltas
		Matrix hiddens_T = Matrix.transpose (hiddens);
		Matrix weights_ho_deltas = Matrix.multiply (gradients, hiddens_T);

		// Update weights by deltas
		weights_ho.add (weights_ho_deltas);
		// Asjust the bias by its deltas (just gradients)
		bias_o.add (gradients);


		//----------
		// Calculate the hidden layer errors
		Matrix weights_ho_transpose = Matrix.transpose(weights_ho);
		Matrix hidden_errors = Matrix.multiply(weights_ho_transpose, output_errors);

		// Calculate hidden gradient
		Matrix hidden_gradient = Matrix.map (hiddens, activeFunc.dfunc);
		hidden_gradient.multiply (hidden_errors);
		hidden_gradient.multiply (learningRate);

		// Calculate input->hidden deltas
		Matrix inputs_T = Matrix.transpose(inputs);
		Matrix weights_ih_deltas = Matrix.multiply (hidden_gradient, inputs_T);

		// Update weights by deltas
		weights_ih.add (weights_ih_deltas);
		// Asjust the bias by its deltas (just gradients)
		bias_h.add (hidden_gradient);


		foreach (float er in output_errors.data) {
			cumError += er * er;
		}
	}


	public void train(float[] input_array, float[] target_array, int batchSize){
		//Generating the Hidden Outputs
		Matrix inputs = Matrix.fromArray (input_array);
		Matrix hiddens = Matrix.multiply (weights_ih, inputs);
		hiddens.add (bias_h);
		//Activation
		hiddens.map (activeFunc.func);

		Matrix outputs = Matrix.multiply (weights_ho, hiddens);
		outputs.add (bias_o);

		// Pass sigmoid as a delegate
		outputs.map (activeFunc.func);

		Matrix targets = Matrix.fromArray (target_array);

		//----------
		// Calculate the output errors
		Matrix output_errors = Matrix.subtract (targets, outputs);


		// Calculate gradient
		Matrix gradients = Matrix.map (outputs, activeFunc.dfunc);
		gradients.multiply (output_errors);

		gradients.multiply (learningRate);

		// Calculate deltas
		Matrix hiddens_T = Matrix.transpose (hiddens);
		Matrix weights_ho_deltas = Matrix.multiply (gradients, hiddens_T);

		// Update weights by deltas
		batchWeights_ho.add (weights_ho_deltas);
		// Asjust the bias by its deltas (just gradients)
		batchBias_o.add (gradients);


		//----------
		// Calculate the hidden layer errors
		Matrix weights_ho_transpose = Matrix.transpose (weights_ho);
		Matrix hidden_errors = Matrix.multiply (weights_ho_transpose, output_errors);

		// Calculate hidden gradient
		Matrix hidden_gradient = Matrix.map (hiddens, activeFunc.dfunc);
		hidden_gradient.multiply (hidden_errors);
		hidden_gradient.multiply (learningRate);

		// Calculate input->hidden deltas
		Matrix inputs_T = Matrix.transpose (inputs);
		Matrix weights_ih_deltas = Matrix.multiply (hidden_gradient, inputs_T);

		// Update weights by deltas
		batchWeights_ih.add (weights_ih_deltas);
		// Asjust the bias by its deltas (just gradients)
		batchBias_h.add (hidden_gradient);

		foreach (float er in output_errors.data) {
			cumError += er * er;
		}

		if (posInBatch == (batchSize - 1)) {
			// Average the batch weights
			batchWeights_ho.divide ((float)batchSize);
			batchWeights_ih.divide ((float)batchSize);

			batchBias_h.divide ((float)batchSize);
			batchBias_o.divide ((float)batchSize);

			//Update the weights
			weights_ho.add (batchWeights_ho);
			weights_ih.add (batchWeights_ih);

			batchBias_h.add (batchBias_h);
			batchBias_o.add (batchBias_o);

			// Reset the batches
			batchWeights_ho.zero ();
			batchWeights_ih.zero ();
			batchBias_h.zero ();
			batchBias_o.zero ();

			posInBatch = 0;
		} else {
			posInBatch++;
		}
	}
}
