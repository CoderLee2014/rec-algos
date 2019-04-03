package robertlee.rec.online.models;

import robertlee.rec.online.data.DataPointMatrix;
import robertlee.rec.online.data.SparseRow;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;

public class FMModel {
	public double[] m_sum, m_sum_sqr;
	public double w0;
	public double[] w;
	public DataPointMatrix v;
	public int num_attribute;
	public boolean k0, k1;
	public int num_factor;
	public int task;

	public double reg0;
	public double regw, regv;

	public double initstdev;
	public double initmean;

	public double max_target;
	public double min_target;

	public FMModel()
	{
		num_factor = 0;
		initmean = 0;
		initstdev = 0.01;
		reg0 = 0.0;
		regw = 0.0;
		regv = 0.0;
		k0 = true;
		k1 = true;
	}

	public void init()
	{
		w0 = 0;
		w = new double[num_attribute];
		v = new DataPointMatrix(num_factor, num_attribute);
		Arrays.fill(w, 0);
		v.init(initmean, initstdev);
		m_sum = new double[num_factor];
		m_sum_sqr = new double[num_factor];
	}

	public double predict(SparseRow x)
	{
		return predict(x, m_sum, m_sum_sqr);
	}

	public double predict(SparseRow x, double[] sum, double[] sum_sqr)
	{
		double result = 0;
		if (k0) {
			result += w0;
		}
		if (k1) {
			for (int i = 0; i < x.getSize(); i++) {
				result += w[x.getData()[i].getId()] * x.getData()[i].getValue();
			}
		}
		for (int f = 0; f < num_factor; f++) {
			sum[f] = 0;
			sum_sqr[f] = 0;
			for (int i = 0; i < x.getSize(); i++) {
				double d = v.get(f,x.getData()[i].getId()) * x.getData()[i].getValue();
				sum[f] = sum[f]+d;
				sum_sqr[f] = sum_sqr[f]+d*d;
			}
			result += 0.5 * (sum[f]*sum[f] - sum_sqr[f]);
		}

		return result;
	}

	public void loadModel(String path) throws Exception
	{
		InputStream is = null;
		DataInputStream dis = null;
		try {
			is = new FileInputStream(path);
			dis = new DataInputStream(is);

			this.k0 = dis.readBoolean();
			this.k1 = dis.readBoolean();
			this.w0 = dis.readDouble();
			this.num_factor = dis.readInt();
			this.num_attribute = dis.readInt();
			this.task = dis.readInt();

			max_target = dis.readDouble();
			min_target = dis.readDouble();

			this.w = new double[this.num_attribute];

			for(int i=0;i<this.num_attribute;i++)
			{
				this.w[i] = dis.readDouble();
			}

			this.m_sum = new double[this.num_factor];
			this.m_sum_sqr = new double[this.num_factor];

			for(int i=0;i<this.num_factor;i++)
			{
				this.m_sum[i] = dis.readDouble();
			}

			for(int i=0;i<this.num_factor;i++)
			{
				this.m_sum_sqr[i] = dis.readDouble();
			}

			this.v = new DataPointMatrix(this.num_factor, this.num_attribute);

			for (int i_1 = 0; i_1 < this.num_factor; i_1++) {
				for (int i_2 = 0; i_2 < this.num_attribute; i_2++) {
					this.v.set(i_1,i_2, dis.readDouble());
				}
			}

		}
		catch(Exception e) {
			System.out.println(e);
		} finally {
			if(dis!=null)
				dis.close();
			if(is!=null)
				is.close();
		}
	}

}