using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq; 
using System.IO;

public class CSVReader : MonoBehaviour 
{
	public TextAsset m_testData; 
	public TextAsset m_trainData; 

	public int testLength;
	public int trainLength;

	private string[] noCommaTest;
	private string[] noCommaTrain;

	public void Start()
	{
		string textGrab = m_testData.text.Replace ("\n", ",");
		noCommaTest = textGrab.Split (',');

		string textGrab2 = m_trainData.text.Replace ("\n", ",");
		noCommaTrain = textGrab2.Split (',');
	}

	public int GetLabel(int index, int state){
		return int.Parse((state == 0) ? noCommaTest [index * 784 + index] : noCommaTrain [index * 784 + index]);
	}

	// pixel array out
	public float[] GetPixels(int index, int state) {
		float[] cols = new float[28 * 28];

		int buffer = index * 784;

		float map = 1.0f / 255;

		for (int i = 0; i < cols.Length; ++i) {
			cols[i] = float.Parse((state == 0) ? noCommaTest[i + buffer + (index + 1)] : noCommaTrain[i + buffer + (index + 1)]) * map;
		}

		return cols;
	}
}