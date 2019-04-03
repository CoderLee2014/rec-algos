package robertlee.rec.online.feature;

import java.util.ArrayList;
import java.util.Map;

public class HashingBucketizer extends FeatureTransformer {
	private Map<String, Double> input;

	public HashingBucketizer(Map<String,Double> tags){
		setInput(tags);
	}

	public ArrayList<Double> transform(){
		return hashing(this.input);
	}

	private int murmurHash(String word, int seed){
		int c1 = 0xcc9e2d51;
		int c2 = 0x1b873593;
		int r1 = 15;
		int r2 = 13;
		int m = 5;
		int n = 0xe6546b64;

		int hash = 12345;

		for (char ch: word.toCharArray()){
			int k = ch;
			k = k * c1;
			k = (k << r1) | (hash >> (32 - r1));
			k = k * c2;

			hash = hash ^ k;
			hash = (hash << r2) | (hash >> (32 - r2));
			hash = hash * m + n;
		}

		hash = hash ^ word.toCharArray().length;
		hash = hash ^ (hash >> 16);
		hash = hash * 0x85ebca6b;
		hash = hash ^ (hash >> 13);
		hash = hash * 0xc2b2ae35;
		hash = hash ^ (hash >> 16);

		return hash;
	}

	private ArrayList<Double> hashing(Map<String, Double> words){
		int BUCKETSIZE = 1000;
		ArrayList<Double> buckets = new ArrayList<Double>(BUCKETSIZE);
		for(int i = 0; i < BUCKETSIZE; i++){
			buckets.add(0.0);
		}
		int seed = 12345;

		for(String word: words.keySet()){
			int bucket = murmurHash(word, seed) % BUCKETSIZE;
			buckets.set(bucket,buckets.get(bucket) + words.get(word));
		}
		System.out.println(buckets);
		return buckets;
	}

	public void setInput(Map<String, Double> input) {
		this.input = input;
	}

}
