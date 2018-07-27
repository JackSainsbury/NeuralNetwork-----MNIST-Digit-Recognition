using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class dataObject{

	public float[] inputs;
	public float[] targets;

	public dataObject(float[] input, float[] target){
		inputs = input;
		targets = target;
	}
}

public class XORNN : MonoBehaviour {

	public Toggle m_00;
	public Toggle m_10;
	public Toggle m_01;
	public Toggle m_11;

	public Toggle m_inA;
	public Toggle m_inB;

	public Text m_output;

	public Text m_logicLabel;

	public InputField m_itterations;
	public InputField m_hiddenNeurons;

	public GameObject m_hiddenContainer;
	public GameObject m_nodeReference;
	public GameObject[] m_nodeInstances;
	public GameObject m_aNode;
	public GameObject m_bNode;
	public GameObject m_outNode;

	public Text m_errorInit;
	public Text m_errorThird;
	public Text m_errorTwoThird;
	public Text m_errorFinal;

	dataObject[] trainingData;

	CSVReader m_texLoad;

	NeuralNetwork nn;

	float[] Guess_array;
	void Start(){
		nn = new NeuralNetwork (2, 4, 1);
		createHiddens ();

		modToggles ();
	}

	public void createHiddens(){
		if (m_nodeInstances != null) {
			for (int i = 0; i < m_nodeInstances.Length; ++i) {
				Destroy(m_nodeInstances[i]);
			}
		}

		int newHiddenLength = int.Parse(m_hiddenNeurons.text);
		float vertBuffer = 75f;

		m_nodeInstances = new GameObject[newHiddenLength];
		for (int i = 0; i < newHiddenLength; ++i) {
			m_nodeInstances [i] = Instantiate (m_nodeReference, m_hiddenContainer.transform.position + new Vector3(0, (vertBuffer * i) - ((newHiddenLength / 2.0f) * vertBuffer), 0), Quaternion.identity);

			m_nodeInstances [i].transform.SetParent (m_hiddenContainer.transform);
		}

		nn = new NeuralNetwork (2, newHiddenLength, 1);
	}

	public void modToggles(){
		bool b00 = m_00.isOn;
		bool b10 = m_10.isOn;
		bool b01 = m_01.isOn;
		bool b11 = m_11.isOn;

		// This is ghastly but works, oh well
		if (b00 && b10 && b01 && b11) {
			m_logicLabel.text = "1";
		} else if (!b00 && !b10 && !b01 && !b11) {
			m_logicLabel.text = "0";
		}else if(!b00 && b10 && b01 && b11){
			m_logicLabel.text = "OR";
		} else if(!b00 && !b10 && !b01 && b11){
			m_logicLabel.text = "AND";
		} else if(b00 && b10 && b01 && !b11){
			m_logicLabel.text = "NAND";
		} else if(b00 && !b10 && !b01 && !b11){
			m_logicLabel.text = "NOR";
		} else if(!b00 && b10 && b01 && !b11){
			m_logicLabel.text = "XOR";
		} else if(b00 && !b10 && !b01 && b11){
			m_logicLabel.text = "EXNOR";
		} else if(!b00 && b10 && !b01 && b11){
			m_logicLabel.text = "A + AND";
		} else if(!b00 && !b10 && b01 && b11){
			m_logicLabel.text = "B + AND";
		} else if(b00 && b10 && !b01 && !b11){
			m_logicLabel.text = "A + NOR";
		} else if(b00 && !b10 && b01 && !b11){
			m_logicLabel.text = "B + NOR";
		} else if(!b00 && b10 && !b01 && !b11){
			m_logicLabel.text = "A";
		} else if(!b00 && !b10 && b01 && !b11){
			m_logicLabel.text = "B";
		} else if(b00 && !b10 && b01 && b11){
			m_logicLabel.text = "!A";
		} else if(b00 && b10 && !b01 && b11){
			m_logicLabel.text = "!B";
		}


		TestNN ();
	}

	public void TrainNN(){
		nn = new NeuralNetwork (2, int.Parse(m_hiddenNeurons.text), 1);

		// XOR training data set

		trainingData = new dataObject[4];
		trainingData[0] = new dataObject(new float[]{0, 1}, new float[]{m_01.isOn ? 1 : 0});
		trainingData[1] = new dataObject(new float[]{1, 0}, new float[]{m_10.isOn ? 1 : 0});
		trainingData[2] = new dataObject(new float[]{1, 1}, new float[]{m_11.isOn ? 1 : 0});
		trainingData[3] = new dataObject(new float[]{0, 0}, new float[]{m_00.isOn ? 1 : 0});

		// train the neural network
		for (int i = 0; i < int.Parse(m_itterations.text); ++i) {
			int randIndex = Random.Range(0, trainingData.Length);
			nn.train (trainingData[randIndex].inputs, trainingData[randIndex].targets);

			if(i == 1){
				m_errorInit.text = "Error " + i.ToString() + " : " + (nn.cumError / i).ToString ();
			}else if(i == int.Parse(m_itterations.text) - 1){
				m_errorFinal.text = "Error " + i.ToString() + " : " + (nn.cumError / i).ToString ();
			}else if(i > 1){
				int thirdIndex = (int)(int.Parse(m_itterations.text)/3.0f);

				if(i % thirdIndex == 0){
					m_errorThird.text = "Error " + i.ToString() + " : " + (nn.cumError / i).ToString ();
				}if(i % (thirdIndex * 2) == 0){
					m_errorTwoThird.text = "Error " + i.ToString() + " : " + (nn.cumError / i).ToString ();
				}
			}


		}
	}

	public void TestNN(){
		float guess = nn.predictNOSoft (new float[]{ m_inA.isOn ? 1 : 0, m_inB.isOn ? 1 : 0 }).data [0, 0];
		m_output.text = guess.ToString () + " (" + Mathf.RoundToInt(guess).ToString() + ")";
	}
		
	public void Update(){
		for (int i = 0; i < m_nodeInstances.Length; ++i) {
			
			Vector3 m_pos = Camera.main.ScreenToWorldPoint (m_nodeInstances [i].transform.position);
			Vector3 a_pos = Camera.main.ScreenToWorldPoint (m_aNode.transform.position);
			Vector3 b_pos = Camera.main.ScreenToWorldPoint (m_bNode.transform.position);
			Vector3 o_pos = Camera.main.ScreenToWorldPoint (m_outNode.transform.position);

			float aWeight = nn.weights_ih.data[i, 0];
			float bWeight = nn.weights_ih.data[i, 1];
			float oWeight = nn.weights_ho.data[0, i];

			Debug.DrawLine (new Vector3(m_pos.x, m_pos.y, 0), new Vector3(a_pos.x, a_pos.y, 0), new Color(aWeight, 0, 0));
			Debug.DrawLine (new Vector3(m_pos.x, m_pos.y, 0), new Vector3(b_pos.x, b_pos.y, 0), new Color(bWeight, 0, 0));
			Debug.DrawLine (new Vector3(m_pos.x, m_pos.y, 0), new Vector3(o_pos.x, o_pos.y, 0), new Color(oWeight, 0, 0));
		}
	}
}
