using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class XORToggle : MonoBehaviour {

	public Text m_label;

	public void toggleValue(){
		m_label.text = GetComponent<Toggle>().isOn ? "1" : "0";
	}

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
