package robertlee.rec.online.feature;

import org.omg.CORBA.Any;

import java.util.ArrayList;

public class FeatureTransformer {
	/*
	* input: 	Map<column_name:string, value:double>
	* output:	float DMatrix
	* Type: OneHotEncoder, HashingBucketizer
	* */
	private ArrayList<Double> res;

	ArrayList<Double> transform(){
		return this.res;
	}
}
