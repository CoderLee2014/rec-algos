package robertlee.rec.online.algos;

import robertlee.rec.online.data.SparseEntry;
import robertlee.rec.online.data.SparseRow;
import robertlee.rec.online.models.FMModel;

import java.util.ArrayList;

public class FM {
	public static void main(String[] args) throws Exception {
		FMModel fm = new FMModel();
		fm.init();
		fm.loadModel("online/src/main/resources/fm_model_2019-02-26_v39");
		ArrayList<SparseEntry> entries = new ArrayList<SparseEntry>();
		SparseEntry ctr = new SparseEntry(10, 0.9);
		SparseEntry age = new SparseEntry(12, 1.0);
		entries.add(ctr);
		entries.add(age);
		SparseRow row = new SparseRow(new SparseEntry[entries.size()]);
		fm.predict(row);
	}

	public void loadTestData(){

	}
}

