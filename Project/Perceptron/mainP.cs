using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class mainP : MonoBehaviour {
	// Visualizing
	public GameObject nodePrefab;

	public Sprite under;
	public Sprite over;
	public Sprite correct;
	public Sprite wrong;

	public InputField m_field;
	public InputField c_field;
	public InputField lr_field;

	private float c = .2f;
	private float m = .3f;

	private float in_learningRate = 0.01f;

	bool lineVis = false;

	float tempC = 0;
	float tempM = 0;

	int trainingIndex = 0;
	static int pointCount = 100;

	Perceptron brain;

	Point[] points = new Point[pointCount];

	GameObject[] pointVisualizers = new GameObject[pointCount];
	GameObject[] pointState = new GameObject[pointCount];

	Graph graph = new Graph (8, 8, Vector3.zero);

	public void updateLine(){
		tempM = float.Parse(m_field.text);
		tempC = float.Parse(c_field.text);

		lineVis = true;

		in_learningRate = float.Parse(lr_field.text);
	}

	void Start () {
		m_field.text = m.ToString ();
		c_field.text = c.ToString ();
		lr_field.text = in_learningRate.ToString ();
		Reload ();
	}

	void Reload() {
		c = tempC;
		m = tempM;
		lineVis = false;
		// create a new perceptron with 3 inputs (x, y, bias)
		brain = new Perceptron(3, in_learningRate);

		bool spawnVisuals = false;

		if (pointVisualizers [0] == null && pointState[0] == null) {
			spawnVisuals = true;
		}

		// generate a set of random points, set their labels (my known input data set)
		for (int i = 0; i < points.Length; ++i) {
			// Generate points with random values
			points [i] = new Point ();
			// Label my training data correctly
			points [i].label = (points [i].y > f (points [i].x)) ? 1 : -1;

			Vector2 mappedPos = graph.mapPoint (points [i]);

			if (spawnVisuals) {
				// visualizers
				pointVisualizers [i] = Instantiate (nodePrefab, new Vector3 (mappedPos.x, mappedPos.y, 0), Quaternion.identity) as GameObject;
				pointState [i] = Instantiate (nodePrefab, new Vector3 (mappedPos.x, mappedPos.y, -0.1f), Quaternion.identity) as GameObject;
			} else {
				pointVisualizers [i].transform.position = new Vector3 (mappedPos.x, mappedPos.y, 0);
				pointState [i].transform.position = new Vector3 (mappedPos.x, mappedPos.y, -0.1f);
			}

			pointVisualizers [i].GetComponent<SpriteRenderer> ().sprite = (points [i].label == 1) ? over : under;
		}
	}

	void Update () {
		if (lineVis) {
			Vector2 pin = graph.mapPoint(new Point (-1, (tempM * -1) + tempC));
			Vector2 pin2 = graph.mapPoint(new Point (1, (tempM * 1) + tempC));
			Debug.DrawLine (new Vector3(pin.x, pin.y, 0), new Vector3(pin2.x, pin2.y, 0), Color.red);
		}

		// draw the graph base
		graph.drawGraph ();

		// draw the base line (the equation f(x) I have used to initially split my known data set into above/below the line
		Vector2 p1 = graph.mapPoint(new Point (-1, f(-1)));
		Vector2 p2 = graph.mapPoint(new Point (1, f(1)));
		Debug.DrawLine (new Vector3(p1.x, p1.y, 0), new Vector3(p2.x, p2.y, 0), Color.blue);
		// draw the varying line, based on my perceptron's weights
		Vector2 p3 = graph.mapPoint(new Point (-1, brain.guessY(-1)));
		Vector2 p4 = graph.mapPoint(new Point (1, brain.guessY(1)));
		Debug.DrawLine (new Vector3(p3.x, p3.y, 0), new Vector3(p4.x, p4.y, 0), Color.blue);

		// Set the visualizers to either correct or wrong, based on a gess for every point (if the line over shoots, correct answers drop back to false and vise versa)
		for(int i = 0; i < points.Length; ++i) {
			// train the brain
			float[] inputs = { points[i].x, points[i].y, points[i].bias };

			// the correct answer
			int target = points[i].label;
			// the guess
			int guess = brain.guess (inputs);
			// update the node visualizers
			if (guess == target) {
				pointState[i].GetComponent<SpriteRenderer> ().sprite = correct;
			} else {
				pointState[i].GetComponent<SpriteRenderer> ().sprite = wrong;
			}
		}

		// Do the training for one point per tick
		Point training = points [trainingIndex];
		float[] trainInputs = {training.x, training.y, training.bias};
		int trainTarget = training.label;
		brain.train(trainInputs, trainTarget);
		trainingIndex ++;
		if(trainingIndex == points.Length){
			trainingIndex = 0;
		}
	}

	// function for line
	float f(float x){
		// y = mx + c
		return m * x + c;
	}

	void OnGUI(){
		if (GUI.Button (new Rect(100,10, 100, 40), "Reset")) {
			Reload ();
		}

		GUI.color = Color.black;

		GUI.Label (new Rect(10,50, 100, 40), "m :");
		GUI.Label (new Rect(10,90, 100, 40), "c :");
		GUI.Label (new Rect(10,130, 100, 40), "Learning Rate :");
	}
}