using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class IMGNet : MonoBehaviour {

	public GameObject[] m_guessOptions;
	GameObject m_guessOption;

	public Texture2D[] m_filters;

	public GUISkin skin;

	public GameObject m_visualizer;
	public GameObject m_visualizerWeights;
	public GameObject m_visualizerOuts;
	public GameObject m_visualizerFCInput_Full;
	public GameObject m_visualizerFCInput;

	public GameObject[] m_featureMapVisualizers;
	public GameObject[] m_filterMapVisualizers;
	public GameObject[] m_poolVisualizers;

	CSVReader m_texLoad;

	NeuralNetwork nn;

	ConvolutionLayer convLayer;

	bool trained = false;

	int graphMode = 0;

	int guess = 0;

	Graph errorDisplay;
	Graph functionGraph;

	float trainTime = 0;

	public int batchSize;

	Matrix[] pooledFeatures;
	float[] Guess_array;

	List<Point> errors;

	// Use this for initialization
	void Start () {
		errors = new List<Point> ();

		convLayer = new ConvolutionLayer (28, 5, 8, 0.01f);

		m_texLoad = GetComponent<CSVReader> ();

		nn = new NeuralNetwork (1152, 81, 2);

		errorDisplay = new Graph (30, 5, new Vector3 (-15, -15, 0));

		functionGraph = new Graph (3, 3, new Vector3 (25, -8, 0));

		for (int i = 0; i < m_featureMapVisualizers.Length; ++i) {
			Texture2D tex = m_filters [i];

			tex.filterMode = FilterMode.Point;

			m_filterMapVisualizers[i].GetComponent<SpriteRenderer>().sprite = Sprite.Create(tex, new Rect(0,0,tex.width,tex.height), new Vector2(0.5f, 0.5f));
		}
	}
		

	public void TrainNN(){
		trained = true;

		float startTrain = Time.realtimeSinceStartup;

		for (int i = 0; i < m_texLoad.trainLength; ++i) {
			// Store errors:
			if ((i % 1 == 0 && i > 0) || i == 1) {
				errors.Add(new Point ((float)i / m_texLoad.trainLength, nn.cumError / i));
			}

			int label = m_texLoad.GetLabel (i, 1);

			float[] targets = new float[]{ 0, 0};
			targets [label] = 1;

			Matrix sample = Matrix.fromArray(m_texLoad.GetPixels (i, 1), 28, 28);

			Matrix[] features = convLayer.Convolve (sample);
			Matrix[] pooledFeatures = convLayer.MaxPool (features);

			float [] FC_Array = new float[pooledFeatures.Length * pooledFeatures[0].getCols() * pooledFeatures[0].getRows()];

			for(int f = 0; f < 8; ++f){
				float[] featureMapArray = pooledFeatures [f].toArray ();

				for(int j = 0; j < featureMapArray.Length; ++j){
					FC_Array [(featureMapArray.Length * f) + j] = featureMapArray [j]; 
				}
			}

			// Pass convonlved to FC, performing forward and backprop
			nn.train(FC_Array, targets, batchSize);
		}

		trainTime = Time.realtimeSinceStartup - startTrain;
	}

	void DrawNum(int index){
		Texture2D tex = new Texture2D (28, 28);

		Color[] cols = new Color[784];

		float[] vals = m_texLoad.GetPixels (index, 0);

		for (int i = 0; i < 784; ++i) {
			cols [i] = new Color (vals [i], vals [i], vals [i]);
		}

		tex.SetPixels (cols);

		tex.Apply ();

		tex.filterMode = FilterMode.Point;

		m_visualizer.GetComponent<SpriteRenderer> ().sprite = Sprite.Create(tex, new Rect(0,0,tex.width,tex.height), new Vector2(0.5f, 0.5f));
	}

	public void GuessNumber(float[] pixels, int label){

		Matrix sample = Matrix.fromArray(pixels, 28, 28);//ConvolutionLayer.ImagePreProcessSubSample (m_texLoad.GetPixels (i, 1));

		Matrix[] features = convLayer.Convolve (sample);

		pooledFeatures = convLayer.MaxPool (features);

		//Messy but temporary
		Guess_array = new float[pooledFeatures.Length * pooledFeatures[0].getCols() * pooledFeatures[0].getRows()];

		for(int f = 0; f < m_featureMapVisualizers.Length; ++f){
			Texture2D activationTexture = new Texture2D (features[f].getRows(), features[f].getCols());
			float[] rawColours = features [f].toArray ();
			Color[] cols = new Color[rawColours.Length];

			Texture2D poolTexture = new Texture2D (pooledFeatures [f].getRows (), pooledFeatures [f].getCols ());
			float[] rawColoursPool = pooledFeatures [f].toArray ();
			Color[] colsPool = new Color[rawColoursPool.Length];

			for(int c = 0; c < rawColours.Length; ++c){
				cols [c] = new Color (rawColours [c], rawColours [c], rawColours [c]);
			}

			for(int c = 0; c < rawColoursPool.Length; ++c){
				colsPool [c] = new Color (rawColoursPool[c], rawColoursPool[c], rawColoursPool[c]);
			}

			activationTexture.SetPixels(cols);
			activationTexture.filterMode = FilterMode.Point;
			activationTexture.Apply ();

			poolTexture.SetPixels(colsPool);
			poolTexture.filterMode = FilterMode.Point;
			poolTexture.Apply ();

			m_featureMapVisualizers[f].GetComponent<SpriteRenderer>().sprite =  Sprite.Create(activationTexture, new Rect(0,0,activationTexture.width,activationTexture.height), new Vector2(0.5f, 0.5f));
			m_poolVisualizers[f].GetComponent<SpriteRenderer>().sprite =  Sprite.Create(poolTexture, new Rect(0,0,poolTexture.width,poolTexture.height), new Vector2(0.5f, 0.5f));

			float[] featureMapArray = pooledFeatures [f].toArray ();

			for(int i = 0; i < featureMapArray.Length; ++i){
				Guess_array [(featureMapArray.Length * f) + i] = featureMapArray [i]; 
			}
		}

		//Test the current training index
		guess = nn.predict (Guess_array);

		if (m_guessOption != null) {
			Destroy (m_guessOption);
		}

		m_guessOption = Instantiate (m_guessOptions [guess], new Vector3(20, 0, 0), Quaternion.identity);

		m_visualizerOuts.GetComponent<SpriteRenderer>().sprite = Sprite.Create(nn.softMaxTexture, new Rect(0,0,nn.softMaxTexture.width, nn.softMaxTexture.height), new Vector3(0.5f, 0.5f));

		m_visualizerWeights.GetComponent<SpriteRenderer> ().sprite = Sprite.Create(nn.activationTexture, new Rect(0,0,1,nn.activationTexture.height * nn.activationTexture.width), new Vector2(0.5f, 0.5f));
	}

	// Update is called once per frame
	void Update () {
		if (Input.GetKeyDown (KeyCode.Tab)) {
			graphMode++;
			if (graphMode > 1) {
				graphMode = 0;
			}
		}

		if (pooledFeatures != null) {
			RaycastHit hit;
			if (Physics.Raycast (Camera.main.ScreenPointToRay (Input.mousePosition), out hit)) {
				if (hit.collider.tag == "PV") {
					m_visualizerFCInput.SetActive (true);

					int index = int.Parse (hit.collider.name [0].ToString ());

					float[] rawWeights = pooledFeatures [index].toArray ();

					Texture2D featureTexture = new Texture2D (1, rawWeights.Length);

					Color[] cols = new Color[rawWeights.Length];

					for (int i = 0; i < rawWeights.Length; ++i) {
						cols [i] = new Color (rawWeights [i], rawWeights [i], rawWeights [i]);
					}

					featureTexture.SetPixels (cols);
					featureTexture.filterMode = FilterMode.Point;
					featureTexture.Apply ();

					m_visualizerFCInput.GetComponent<SpriteRenderer> ().sprite = Sprite.Create (featureTexture, new Rect (0, 0, featureTexture.width, featureTexture.height), new Vector2 (0.5f, 0.5f));

					Debug.DrawLine (hit.collider.transform.position + new Vector3(1.2f, 1.2f, 0), m_visualizerFCInput_Full.transform.position + new Vector3(0, 12, 0) - new Vector3(0, 3 * index, 0));
					Debug.DrawLine (hit.collider.transform.position + new Vector3(1.2f, -1.2f, 0), m_visualizerFCInput_Full.transform.position + new Vector3(0, 12, 0) - new Vector3(0, 3 * (index + 1), 0));

					Debug.DrawLine (m_visualizerFCInput_Full.transform.position + new Vector3(0, 12, 0) - new Vector3(0, 3 * index, 0), m_visualizerFCInput.transform.position + new Vector3(0, 12, 0));
					Debug.DrawLine (m_visualizerFCInput_Full.transform.position + new Vector3(0, 12, 0) - new Vector3(0, 3 * (index + 1), 0), m_visualizerFCInput.transform.position + new Vector3(0, -12, 0));
				}
			} else {
				m_visualizerFCInput.SetActive (false);
			}

			if (Guess_array != null) {
				Texture2D featureTexture = new Texture2D (1, Guess_array.Length);

				Color[] cols = new Color[Guess_array.Length];

				for (int i = 0; i < Guess_array.Length; ++i) {
					cols [i] = new Color (Guess_array [i], Guess_array [i], Guess_array [i]);
				}

				featureTexture.SetPixels (cols);
				featureTexture.filterMode = FilterMode.Point;
				featureTexture.Apply ();

				m_visualizerFCInput_Full.GetComponent<SpriteRenderer> ().sprite = Sprite.Create (featureTexture, new Rect (0, 0, featureTexture.width, featureTexture.height), new Vector2 (0.5f, 0.5f));
			}
		}
			
		if (trained) {
			errorDisplay.drawGraph ();

			for (int i = 0; i < errors.Count; ++i) {
				int next = i + 1;

				if (next == errors.Count) {
					break;
				}

				if (graphMode == 0) {
					if (i % (batchSize - 1) == 0) {
						Debug.DrawLine (errorDisplay.graphOrigin + new Vector3 (errors [i].x * errorDisplay.xAxis, 0, 0), errorDisplay.graphOrigin + new Vector3 (errors [i].x * errorDisplay.xAxis, errors [i].y * errorDisplay.yAxis, 0), Color.green);
					}
					Debug.DrawLine (errorDisplay.graphOrigin + new Vector3 (errors [i].x * errorDisplay.xAxis, errors [i].y * errorDisplay.yAxis, 0), errorDisplay.graphOrigin + new Vector3 (errors [next].x * errorDisplay.xAxis, errors [next].y * errorDisplay.yAxis, 0), Color.red);
				} else {
					if (i % (batchSize - 1) == 0) {
						Debug.DrawLine (errorDisplay.graphOrigin + new Vector3 (errors [i].x * errorDisplay.xAxis, 0, 0), errorDisplay.graphOrigin + new Vector3 (errors [i].x * errorDisplay.xAxis, errors [i].y * errorDisplay.yAxis, 0), Color.green);
					}
				}
			}
		} else {
			functionGraph.drawGraph ();

			// draw graph test
			for(int i = -60; i < 60; ++i){
				float next = (i + 1);

				float FuncI = nn.activeFunc.func (i/20.0f);
				float FuncN = nn.activeFunc.func (next/20.0f);

				Debug.DrawLine (functionGraph.graphOrigin + new Vector3 ((float)i/20.0f, FuncI * functionGraph.yAxis, 0), functionGraph.graphOrigin + new Vector3 (next/20.0f, FuncN * functionGraph.yAxis, 0), Color.green);
				Debug.DrawLine (functionGraph.graphOrigin + new Vector3 ((float)i/20.0f, nn.activeFunc.dfunc (FuncI) * functionGraph.yAxis, 0), functionGraph.graphOrigin + new Vector3 (next/20.0f, nn.activeFunc.dfunc (FuncN) * functionGraph.yAxis, 0), Color.red);

			}
		}
	}


	void OnGUI(){
		skin.label.fontSize = 20;
		GUI.skin = skin;

		if (!trained) {
			GUI.color = Color.red;
			GUI.Label (new Rect (10, 10, 100, 100), "Untrained.");
		} else {
			GUI.color = Color.green;
			GUI.Label (new Rect (10, 10, 100, 100), "Trained.");
			GUI.Label (new Rect (10, 40, 100, 100), "Train time: " + trainTime.ToString());
		}

		GUI.skin = skin;
		GUI.color = Color.black;

		skin.label.fontSize = 50;
	}
}
