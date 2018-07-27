using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Point {

	public float x;
	public float y;
	public float bias = 1;
	public int label;

	public Point(float x_, float y_){
		x = x_;
		y = y_;
	}

	public Point(){
		x = Random.Range (-1.0f, 1.0f);
		y = Random.Range (-1.0f, 1.0f);
	}
}
