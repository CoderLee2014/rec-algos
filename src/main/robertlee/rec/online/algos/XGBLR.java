package robertlee.rec.online.algos;

import ml.combust.mleap.core.types.*;
import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.runtime.frame.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class XGBLR {
	private String modelPath;
	private StructType dataSchema;

	private Transformer model;

	public XGBLR(String modelPath, StructType dataSchema){
		this.modelPath = modelPath;
		this.dataSchema = dataSchema;
	}

	public XGBLR(String modelPath){
		this.modelPath = modelPath;
		loadModel();
	}

	public void loadModel(){
		MleapContext mleapContext = new ContextBuilder().createMleapContext();
		BundleBuilder bundleBuilder = new BundleBuilder();
		this.model = bundleBuilder.load(new File(this.modelPath), mleapContext).root();
		System.out.println(this.model.model());
	}

	public Row forecast(Row features){
		if (this.model == null){
			loadModel();
		}
		if (features == null){
			System.err.println("features is null");
			return null;
		}

		LeapFrameBuilder builder = new LeapFrameBuilder();

		ArrayList<Row> rows = new ArrayList<>();
		rows.add(features);
		DefaultLeapFrame frame = builder.createFrame(dataSchema, rows);
		System.out.println(frame);
		DefaultLeapFrame result = this.model.transform(frame).get();
		return result.dataset().head();
	}

	public static void main(String[] args){
		LeapFrameBuilder builder = new LeapFrameBuilder();
		List<StructField> fields = new ArrayList();
		fields.add(builder.createField("frequent_city", builder.createString()));
		fields.add(builder.createField("device_model", builder.createString()));
		fields.add(builder.createField("xgb_leaf", builder.createTensor(builder.createBasicDouble(),new ArrayList<Integer>(Arrays.asList((Integer)1280)),true)));

		StructType schema = builder.createSchema(fields);
		XGBLR server = new XGBLR("online/src/main/resources/tmp_lr_pipeline_2019-01-04_v4.bundle.zip",schema);

		String frequent_city = "863301";
		String device_model = "HUAWEI MLA-AL10";
		Vector xgb_leaf = Vectors.dense(new double[1280]);
		Row features = builder.createRow(frequent_city,device_model, xgb_leaf);

		Row result = server.forecast(features);
		for(int i = 0 ; i < result.size(); i++){
			System.out.println(result.get(i));
		}
	}
}
