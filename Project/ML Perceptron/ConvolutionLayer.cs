using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ConvolutionLayer {

	int numFilters;
	int height_in;
	int width_in;
	int filterSize;

	Matrix[] m_weights;
	float[] m_biases;

	int outputDim1;
	int outputDim2;

	Matrix[] zValues;
	Matrix[] output;

	Vector2[] poolIndices;

	public ConvolutionLayer(int _inputShape, int _filterSize, int _numFilters, float _bias){
		height_in = _inputShape;
		width_in = _inputShape;

		filterSize = _filterSize;

		numFilters = _numFilters;

		m_weights = new Matrix[numFilters];

		for(int j = 0; j < numFilters; ++j) {
			m_weights[j] = new Matrix (filterSize, filterSize);
		}

		m_biases = new float[numFilters];

		for (int i = 0; i < numFilters; ++i) {
			m_biases [i] = Random.Range(-_bias, _bias);
		}

		outputDim1 = (height_in - filterSize) + 1;
		outputDim2 = (width_in - filterSize) + 1;

		zValues = new Matrix [numFilters];
		output = new Matrix [numFilters];

		// 3d shape initially
		for (int i = 0; i < numFilters; ++i) {
			zValues [i] = new Matrix (outputDim1, outputDim2);
			output [i] = new Matrix (outputDim1, outputDim2);
		}

		LoadFilters ();
	}

	public void LoadFilters(){
		for (int i = 0; i < numFilters; ++i) {
			string path = "Assets/Filters/filter" + i.ToString () + ".txt";
			StreamReader reader = new StreamReader (path);

			string lines = reader.ReadToEnd ();

			string[] splitLines = lines.Split (',');

			for(int x = 0; x < filterSize; ++x){
				for(int y = 0; y < filterSize; ++y){
					m_weights[i].data[x, y] = float.Parse (splitLines [(y * filterSize) + x].Replace("\n", ""));
				}
			}

			reader.Close ();
		}
	}

	public Matrix[] Convolve(Matrix inputNeurons){
		// Do convolution processing for a set of input neurons (raw image/pool sub sampled)

		// For each filters
		for (int j = 0; j < numFilters; ++j) {
			// Create 2 matrices with numfilters rows and outDims square cols

			zValues [j].zero();
			output [j].zero();

			for (int i = 0; i < outputDim1; ++i) {
				for (int k = 0; k < outputDim2; ++k) {
					for (int _x = 0; _x < filterSize; ++_x) {
						for (int _y = 0; _y < filterSize; ++_y) {
							zValues [j].data [i, k] += (inputNeurons.data [i + (_x), k + (_y)] * 2 - 1) * (m_weights [j].data [_x, _y] * 2 - 1);
						}
					}
					output [j].data [i, k] = (NeuralNetwork.relu.func(zValues [j].data [i, k]) / 25.0f) + m_biases [j]; 
				}
			}
		}
			
		return output;
	}

	public Matrix[] MeanPool(Matrix[] featureMaps){
		Matrix[] pooledFeatures = new Matrix[featureMaps.Length];

		for (int f = 0; f < featureMaps.Length; ++f) {

			pooledFeatures[f] = new Matrix (featureMaps [f].getRows () / 2, featureMaps [f].getCols ()/ 2);

			for (int i = 0; i < featureMaps [f].getRows () - 1; i += 2) {
				for (int j = 0; j < featureMaps [f].getCols () - 1; j += 2) {
					for (int _x = 0; _x < 2; ++_x) {
						for (int _y = 0; _y < 2; ++_y) {
							pooledFeatures[f].data[i/2, j/2] += featureMaps[f].data[i + _x, j + _y];
						}
					}
				}
			}
			pooledFeatures [f].divide (4);
		}

		return pooledFeatures;
	}

	public Matrix[] MaxPool(Matrix[] featureMaps){
		Matrix[] pooledFeatures = new Matrix[featureMaps.Length];

		int poolLength = (featureMaps [0].getRows () / 2) * (featureMaps [0].getCols () / 2);

		// Pool indices need to be saved when performing max pooling, store in a big vector array
		poolIndices = new Vector2[featureMaps.Length * poolLength];

		for (int f = 0; f < featureMaps.Length; ++f) {

			pooledFeatures[f] = new Matrix (featureMaps [f].getRows () / 2, featureMaps [f].getCols ()/ 2);

			for (int i = 0; i < featureMaps [f].getRows () - 1; i += 2) {
				for (int j = 0; j < featureMaps [f].getCols () - 1; j += 2) {
					float max = 0;
					Vector2 maxIndex = Vector2.zero;
					for (int _x = 0; _x < 2; ++_x) {
						for (int _y = 0; _y < 2; ++_y) {
							if (featureMaps [f].data [i + _x, j + _y] > max) {
								// Get the max value in the region
								max = featureMaps [f].data [i + _x, j + _y];

								maxIndex = new Vector2(i + _x, j + _y);
							}
						}
					}
					// Put the max feature into the pooled map
					pooledFeatures [f].data[i/2, j/2] = max;
					// Store the i, j position in the original feature map for backprop
					poolIndices [((f * poolLength) + ((i / 2) * (featureMaps [f].getRows ()/2)) + (j / 2))] = maxIndex;
				}
			}
		}

		return pooledFeatures;
	}

	public void convBackProp () {
		
	}

	// Pre processing for the first input neuron set into the conv layer
	// Grab a randome 24x24 sample of a 28 * 28 image
	public static Matrix ImagePreProcessSubSample(float[] rawPixels){
		// make a matrix, temporarily (messy and lossy logic)
		Matrix rawMat = Matrix.fromArray (rawPixels, 28, 28);
		// the sub matrix
		Matrix sample = new Matrix (24, 24);

		// Randomly offset the 24x24 in the 28x28
		int xOff = Random.Range (0, 4);
		int yOff = Random.Range (0, 4);

		for (int i = 0; i < 24; ++i) {
			for (int j = 0; j < 24; ++j) {
				sample.data [i, j] = rawMat.data [i + xOff, j + yOff];
			}
		}

		return sample;
	}
}
