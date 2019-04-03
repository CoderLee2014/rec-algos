package robertlee.rec.online.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OneHotEncoder extends FeatureTransformer {
	private Map<String, Integer> labels;
	private String input;
	private ArrayList<Double> output;

	public OneHotEncoder(ArrayList<String> labels){
		setLabels(labels);
	}

	public ArrayList<Double> transform(String input){
		setInupt(input);
		this.encoding();
		System.out.println("encoded as: "+this.output);
		return this.output;
	}

	public void encoding(){
		this.output = new ArrayList<Double>(this.labels.size());
		for(int i = 0; i < this.labels.size(); i++){
			this.output.add(0.0);
		}
		this.output.set(this.labels.get(this.input),1.0);
	}

	public void setLabels(ArrayList<String> labels) {
		int i = 0;
		this.labels = new HashMap<String, Integer>();
		for(String item: labels){
			this.labels.put(item, i);
			i += 1;
		}
	}

	public void setInupt(String inupt) {
		this.input = inupt;
	}
}
