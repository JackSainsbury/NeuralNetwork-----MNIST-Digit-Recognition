using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class fileProcessed {
	public float[] m_pixels;
	public int m_label;

	public fileProcessed(Color[] pixels, int label){
		m_label = label;
		m_pixels = new float[pixels.Length];

		// Push to float array
		for (int i = 0; i < 28; ++i) {
			for (int j = 0; j < 28; ++j) {
				m_pixels [((28*28) - 1) - (j*28 + i)] = 1 - pixels [((27 - i)*28 + j)].r;
			}
		}
	}
}

public class DataSetLabeller : MonoBehaviour {

	Plane m_Plane;

	Vector3 m_drawOrigin;

	float m_drawDimension = 5;

	public GameObject m_drawVisQuad;

	Texture2D m_drawTexture;

	float drawSubStep = 0;

	private bool m_drawChange = false;

	fileProcessed DrawFile;

	List<fileProcessed> FilesToWrite;

	private string m_label = "0";

	// Use this for initialization
	void Start () {
		FilesToWrite = new List<fileProcessed> ();

		m_drawTexture = new Texture2D (28, 28);
		m_drawTexture.filterMode = FilterMode.Point;

		Color[] cols = new Color[28*28];

		for (int i = 0; i < (28 * 28); ++i) {
			cols [i] = Color.white;
		}

		m_drawTexture.SetPixels (cols);
		m_drawTexture.Apply ();

		drawSubStep = 28.0f / (m_drawDimension * 2);

		m_Plane = new Plane(Vector3.forward, Vector3.zero);

		m_drawOrigin = new Vector3(14, 0, 0);

		m_drawVisQuad.transform.position = -m_drawOrigin;
		float size = ((m_drawDimension * 2) / 28.0f) * 100;
		m_drawVisQuad.transform.localScale = new Vector3 (size, -size, size);

		m_drawVisQuad.GetComponent<SpriteRenderer> ().sprite = Sprite.Create (m_drawTexture, new Rect (0, 0, m_drawTexture.width, m_drawTexture.height), new Vector2 (0.5f, 0.5f));
	}

	void WriteNewCSV(List<fileProcessed> file){
		int files = file.Count;

		using(StreamWriter w = new StreamWriter("Assets/Test/train.csv"))
		{
			for (int i = 0; i < files; ++i) {
				int index = Random.Range (0, file.Count);

				string line = file[index].m_label.ToString();

				for (int j = 0; j < 784; ++j) {
					line = string.Format ("{0},{1}", line, ((int)(file[index].m_pixels[j] * 255)).ToString()); 
				}

				w.WriteLine (line);
				w.Flush ();

				file.RemoveAt (index);
			}
		}
	}
	
	// Update is called once per frame
	void Update () {

		Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

		float enter = 0.0f;

		if (Input.GetMouseButton (0)) {
			if (m_Plane.Raycast (ray, out enter)) {
				Vector3 hitPoint = ray.GetPoint (enter) + m_drawOrigin;
				if (hitPoint.x > -m_drawDimension && hitPoint.x < m_drawDimension) {
					Color[] cols = m_drawTexture.GetPixels ();

					int _x = Mathf.RoundToInt ((hitPoint.x + m_drawDimension) * drawSubStep);
					int _y = Mathf.RoundToInt ((hitPoint.y + m_drawDimension) * drawSubStep);

					_x = (_x == 28) ? 27 : _x;
					_y = (_y == 28) ? 27 : _y;


					for (int i = -1; i < 2; ++i) {
						for (int j = -1; j < 2; ++j) {
							if (i == 0 && j == 0 && _x >= 0 && _y >= 0 && _x < 28 && _y < 28) {
								cols [_x * 28 + _y] = Color.black;
							} else {
								if (_x + i >= 0 && _y + j >= 0 && _x + i < 28 && _y + j < 28) {
									Color cur = cols [(_x + i) * 28 + _y + j];
									cols [(_x + i) * 28 + _y + j] = new Color (cur.r - .3f, cur.g - .3f, cur.b - .3f);
								}
							}
						}
					}

					m_drawTexture.SetPixels (cols);
					m_drawTexture.Apply ();
					m_drawVisQuad.GetComponent<SpriteRenderer> ().sprite = Sprite.Create (m_drawTexture, new Rect (0, 0, m_drawTexture.width, m_drawTexture.height), new Vector2 (0.5f, 0.5f));
					m_drawChange = true;
				}
			}
		} else if(m_drawChange) {
			//For live updating the draw area
			string parseString = m_label.Replace("\n","");
			DrawFile = new fileProcessed(m_drawTexture.GetPixels(), int.Parse(parseString));
			Camera.main.GetComponent<IMGNet> ().GuessNumber (DrawFile.m_pixels, DrawFile.m_label);
			m_drawChange = false;
		}

		if (Input.GetKeyDown (KeyCode.Backspace)) {
			Color[] cols = new Color[28*28];

			for (int i = 0; i < (28 * 28); ++i) {
				cols [i] = Color.white;
			}

			m_drawTexture.SetPixels (cols);
			m_drawTexture.Apply ();
		}

		// for saving to file
		if (Input.GetKeyDown (KeyCode.Return)) {
			FilesToWrite.Add( new fileProcessed(m_drawTexture.GetPixels(), int.Parse(m_label)));

			Color[] cols = new Color[28*28];

			for (int i = 0; i < (28 * 28); ++i) {
				cols [i] = Color.white;
			}

			m_drawTexture.SetPixels (cols);
			m_drawTexture.Apply ();
		}

		Debug.DrawLine (new Vector3 (-m_drawDimension, m_drawDimension, 0) - m_drawOrigin, new Vector3 (m_drawDimension, m_drawDimension, 0) - m_drawOrigin, Color.red);
		Debug.DrawLine (new Vector3 (-m_drawDimension, m_drawDimension, 0) - m_drawOrigin, new Vector3 (-m_drawDimension, -m_drawDimension, 0) - m_drawOrigin, Color.red);
		Debug.DrawLine (new Vector3 (-m_drawDimension, -m_drawDimension, 0) - m_drawOrigin, new Vector3 (m_drawDimension, -m_drawDimension, 0) - m_drawOrigin, Color.red);
		Debug.DrawLine (new Vector3 (m_drawDimension, m_drawDimension, 0) - m_drawOrigin, new Vector3 (m_drawDimension, -m_drawDimension, 0) - m_drawOrigin, Color.red);
	}
		
	public void Clear(){
		Color[] cols = new Color[28*28];

		for (int i = 0; i < (28 * 28); ++i) {
			cols [i] = Color.white;
		}

		m_drawTexture.SetPixels (cols);
		m_drawTexture.Apply ();
	}

	void OnGUI(){
		GUI.Label (new Rect(10,10, 100, 100), FilesToWrite.Count.ToString());

		m_label = GUI.TextField (new Rect (Screen.width / 6.5f, Screen.height / 1.4f, 100, 50), m_label);

		if(GUI.Button(new Rect(50, Screen.height - 50, 100, 50), "Write CSV")){
			if (FilesToWrite.Count > 0) {
				WriteNewCSV (FilesToWrite);
			}
		}
	}
}
