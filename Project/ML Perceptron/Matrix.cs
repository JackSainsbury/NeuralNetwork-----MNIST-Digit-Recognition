using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Matrix class, for understanding the operations (swap for something more efficient later)
public class Matrix {
	int rows;
	int cols;
	public float[,] data;

	// Map function delegate
	public delegate float mapFunc (float matVal);

	public Matrix(int rows_, int cols_){
		rows = rows_;
		cols = cols_;

		data = new float[rows, cols];

		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] = 0;
			}
		}
	}

	public int getRows(){
		return rows;
	}
		
	public int getCols(){
		return cols;
	}

	//------------------------------------------------------------
	// Scalar multiplication
	public void multiply(float n){
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] *= n;
			}
		}
	}

	// Matrix element-wise multiplication (HadamardProduct)
	public void multiply(Matrix other){
		// Dimensions are not the same
		if(other.rows != rows || other.cols != cols){
			Debug.Log ("Dimensions of A must match Dimensions of B");
			return;
		}

		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data [i, j] *= other.data [i, j];
			}
		}
	}

	// Cross multiply
	public static Matrix multiply(Matrix a, Matrix b){
		// 'Dot sensitive' Dimension is not the same
		if(a.cols != b.rows){
			Debug.Log ("Columns of A must match Rows of B");
			return null;
		}

		Matrix result = new Matrix (a.rows, b.cols);

		for(int i = 0; i < result.rows; ++i){
			for(int j = 0; j < result.cols; ++j){
				//Dot products of values in col
				float sum = 0;
				for (int k = 0; k < a.cols; ++k) {
					sum += a.data [i, k] * b.data [k, j];
				}

				result.data [i, j] = sum;
			}
		}

		return result;
	}
		
	//------------------------------------------------------------
	// Zero Matrix
	public void zero(){
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] = 0;
			}
		}
	}

	//------------------------------------------------------------
	// Scalar addition
	public void add(float n){
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] += n;
			}
		}
	}

	// Matrix element-wise addition
	public void add(Matrix other){
		// Dimensions are not the same
		if(other.rows != rows || other.cols != cols){
			Debug.Log ("Dimensions of A must match Dimensions of B");
			return;
		}

		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] += other.data[i,j];
			}
		}
	}

	// static Matrix element-wise addition
	public static Matrix add(Matrix a, Matrix b){
		// Dimensions are not the same
		if(a.rows != b.rows || a.cols != b.cols){
			Debug.Log ("Dimensions of A must match Dimensions of B");
			return null;
		}

		Matrix returnMat = a;

		for(int i = 0; i < a.rows; ++i){
			for(int j = 0; j < a.cols; ++j){
				returnMat.data[i,j] += b.data[i,j];
			}
		}

		return returnMat;
	}

	//------------------------------------------------------------
	// Scalar division
	// Matrix element-wise addition
	public void divide(float divisor){
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] /= divisor;
			}
		}
	}

	//------------------------------------------------------------
	// Reshape a matrix to the supplied dimensions
	public static Matrix reShape(Matrix m, int _rows, int _cols){
		// Check if the resize is appropriate
		if (m.rows * m.cols != _rows * _cols) {
			Debug.Log ("Dimensions of A must match Dimensions of B during reshape");
			return null;
		}

		Matrix mat = new Matrix (_rows, _cols);

		float[] orig = m.toArray ();

		for (int i = 0; i < _rows; ++i) {
			for (int j = 0; j < _cols; ++j) {
				mat.data [i, j] = orig [j * _rows + i];
			}
		}

		return mat;
	}

	//------------------------------------------------------------
	// Return a new matrix a-b
	public static Matrix subtract(Matrix a, Matrix b){
		// Dimensions are not the same
		if(a.rows != b.rows || a.cols != b.cols){
			Debug.Log ("Dimensions of A must match Dimensions of B during subtract");
			return null;
		}

		Matrix result = new Matrix (a.rows, a.cols);

		for(int i = 0; i < a.rows; ++i){
			for(int j = 0; j < a.cols; ++j){
				result.data[i,j] = a.data[i,j] - b.data[i,j];
			}
		}

		return result;
	}

	//------------------------------------------------------------
	// Transposition
	public void transposed(){
		Matrix result = new Matrix (cols, rows);

		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				result.data[j,i] = data[i,j];
			}
		}

		rows = result.rows;
		cols = result.cols;
		data = result.data;
	}

	// Transposition
	public static Matrix transpose(Matrix a){
		Matrix result = new Matrix (a.cols, a.rows);

		for(int i = 0; i < a.rows; ++i){
			for(int j = 0; j < a.cols; ++j){
				result.data[j,i] = a.data[i,j];
			}
		}
		return result;
	}

	//------------------------------------------------------------
	// Randomize all elements from 0-9
	public void randomize(){
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data [i, j] = Random.Range(-1.0f, 1.0f);
			}
		}
	}

	//------------------------------------------------------------
	//Apply a function to every element of matrix
	public void map (mapFunc delegateFunction)
	{
		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				data[i,j] = delegateFunction (data[i,j]);
			}
		}
	}
		
	//Apply a function to every element of matrix
	public static Matrix map (Matrix m, mapFunc delegateFunction)
	{
		Matrix result = new Matrix (m.rows, m.cols);

		for(int i = 0; i < m.rows; ++i){
			for(int j = 0; j < m.cols; ++j){
				result.data[i,j] = delegateFunction (m.data[i,j]);
			}
		}

		return result;
	}

	//------------------------------------------------------------
	// Create Matrix from Array
	public static Matrix fromArray(float[] arr){

		Matrix result = new Matrix (arr.Length, 1);

		for(int i = 0; i < arr.Length; ++i){
			result.data[i,0] = arr[i]; 
		}

		return result;
	}

	// Create Matrix from Array with dimension test
	public static Matrix fromArray(float[] arr, int dRows, int dCols){
		if (dRows * dCols != arr.Length) {
			Debug.Log ("Rows * Cols must equal array length");
			return null;
		}

		Matrix result = new Matrix (dRows, dCols);

		for(int i = 0; i < dRows; ++i){
			for(int j = 0; j < dCols; ++j){
				result.data[i,j] = arr[j * dCols + i]; 
			}
		}

		return result;
	}

	//------------------------------------------------------------
	public float[] toArray(){
		float[] result = new float[rows * cols];

		for(int i = 0; i < rows; ++i){
			for(int j = 0; j < cols; ++j){
				result [(j * cols) + i] = data [i, j];
			}
		}

		return result;
	}

	//------------------------------------------------------------
	public void print(){
		for(int i = 0; i < rows; ++i){

			string line = "";

			for(int j = 0; j < cols; ++j){
				line += data [i, j].ToString () + ", ";
			}

			Debug.Log (line);
		}
	}

	public void print(string label){
		for(int i = 0; i < rows; ++i){

			string line = label + " : ";

			for(int j = 0; j < cols; ++j){
				line += data [i, j].ToString () + ", ";
			}

			Debug.Log (line);
		}
	}
}
