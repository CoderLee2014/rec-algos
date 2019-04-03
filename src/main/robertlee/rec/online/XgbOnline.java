package robertlee.rec.online;


import robertlee.rec.online.feature.HashingBucketizer;
import robertlee.rec.online.feature.OneHotEncoder;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


public class XgbOnline {
	public static void main(String[] args) throws IOException, XGBoostError, URISyntaxException {

		FileSystem fs = FileSystem.get(new URI("viewfs://hadoop/data/robert_rec/model/"),new Configuration());
		FSDataInputStream nativeModelPath = fs.open(new Path("viewfs://hadoop/data/robert_rec/model/xgb_model_"  + args[0]));

		Booster booster2 = XGBoost.loadModel(nativeModelPath);
		System.out.println("Model loaded succeeded.");
		String data = "868039030644792\t108943426348\t0\t0\t0\t0\t0\t{}\t{}\t2018-12-02\n";
		int i = 0;
		Integer[] tags = new Integer[] {7, 8, 9, 10, 13, 14};
		ArrayList<Integer> tags_index = new ArrayList<Integer>(Arrays.asList(tags));
		Integer[] oneHot = new Integer[] {11,15,49,50};
		ArrayList<Integer> oneHot_index = new ArrayList<Integer>(Arrays.asList(oneHot));
		Integer[] excludes = new Integer[] {0,1,48,51,52,53};
		ArrayList<Integer> excludes_index = new ArrayList<Integer>(Arrays.asList(excludes));

		ArrayList<String> labels_feed_type = new ArrayList<String>(Arrays.asList("1","3","9"));
		ArrayList<String> labels_has_title = new ArrayList<String>(Arrays.asList("false","true"));
		ArrayList<String> labels_r_source = new ArrayList<String>(Arrays.asList("-1","0","1","2","3"));
		ArrayList<String> labels_age = new ArrayList<String>(Arrays.asList("-1","0","1","2","3","4","5","6"));

		ArrayList<Double> res = new ArrayList<Double>();
		OneHotEncoder oneHotEncoder_feed_type = new OneHotEncoder(labels_feed_type);
		OneHotEncoder oneHotEncoder_has_title = new OneHotEncoder(labels_has_title);
		OneHotEncoder oneHotEncoder_r_source = new OneHotEncoder(labels_r_source);
		OneHotEncoder oneHotEncoder_age = new OneHotEncoder(labels_age);
		for(String col: data.split("\t")){
			if(excludes_index.contains(i)){
				i += 1;
				continue;
			}
			if(tags_index.contains(i)){
				String[] scores = col.substring(1,col.length()-1).split(",");
				HashMap map = new HashMap();
				for(String score: scores){
					System.out.println(score);
					if(score.equals(""))
						break;
					String[] tmp = score.split(":");
					if(tmp.length == 1)
						map.put(score.split(":")[0], 1.0);
					else
						map.put(tmp[0], Double.valueOf(tmp[1]));
				}
				System.out.println(map);
				res.addAll(new HashingBucketizer(map).transform());
			}
			if (oneHot_index.contains(i)){
				if(i == 11){
					res.addAll(oneHotEncoder_feed_type.transform(col));
				}else if(i == 15){
					res.addAll(oneHotEncoder_has_title.transform(col));
				}else if(i == 49){
					res.addAll(oneHotEncoder_r_source.transform(col));
				}else if(i == 50){
					res.addAll(oneHotEncoder_age.transform(col));
				}
			}
			i += 1;
		}
		float[] test = new float[res.size()];
		i = 0;
		for(Double f: res){
			test[i] = f.floatValue();
			i += 1;
		}
		DMatrix testMat2 = new DMatrix(test,1,1,0.0f);
		float[][] predicts2 = booster2.predict(testMat2);
		for(float[] pred: predicts2){
			for(float dd : pred) {
				System.out.println(dd);
			}
		}
	}
}
