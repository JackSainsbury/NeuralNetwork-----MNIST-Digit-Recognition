using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Graph {
	//Graph axis
	public int xAxis = 8;
	public int yAxis = 8;

	public Vector3 graphOrigin;

	public Graph(int x, int y, Vector3 origin){
		xAxis = x;
		yAxis = y;

		graphOrigin = origin;
	}

	public Vector2 mapPoint(Point p){
		return new Vector2 (
			(p.x - -1) * (xAxis - -xAxis) / (1 - -1) + -xAxis, 
			(p.y - -1) * (yAxis - -yAxis) / (1 - -1) + -yAxis
		);
	}

	void drawAxis(Vector3 start, Vector3 end, bool hor){
		Debug.DrawLine (graphOrigin + start, graphOrigin + end, Color.black);

		int lineMag = Mathf.RoundToInt((start - end).magnitude);

		for (int i = 0; i < lineMag/2; ++i) {
			int off = (i + 1);

			if (!hor) {
				Debug.DrawLine (graphOrigin + new Vector3 (-.1f, off, 0), graphOrigin + new Vector3 (.1f, off, 0), Color.black);
				Debug.DrawLine (graphOrigin + new Vector3 (-.1f, -off, 0), graphOrigin + new Vector3 (.1f, -off, 0), Color.black);
			} else {
				Debug.DrawLine (graphOrigin + new Vector3 (off, -.1f, 0), graphOrigin + new Vector3 (off, .1f, 0), Color.black);
				Debug.DrawLine (graphOrigin + new Vector3 (-off, -.1f, 0), graphOrigin + new Vector3 (-off, .1f, 0), Color.black);
			}
		}
	}

	public void drawGraph(){
		drawAxis (new Vector3(0, yAxis, 0), new Vector3(0, -yAxis, 0), false);
		drawAxis (new Vector3(xAxis, 0, 0), new Vector3(-xAxis, 0, 0), true);
	}
}
