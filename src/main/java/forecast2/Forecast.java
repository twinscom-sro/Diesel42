package forecast2;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import predictorTasks.BacktestingProcessor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Forecast {

    public static void main(String[] args) throws IOException {
        String netFile = "c:/_db/nets_120/CAT_14A__buys.txt";
        String tsFile = "c:/_db/ts_120/CAT_full.txt";

        MultiLayerNetwork net = MultiLayerNetwork.load(new File(netFile), true);

        Forecast f = new Forecast(20000, 120);
        DataSet ds = f.loadDataSet( tsFile, 2016, 2023 );

        for( int col=-1; col<24; col++ ) {
            INDArray vector1 = Nd4j.create(f.numDays, f.VECTOR_SIZE);

            for( int d=0; d<f.numDays; d++ ) {
                for( int k=0; k<f.VECTOR_SIZE; k++ ){
                    double value = f.inputs[d][k];
                    vector1.putScalar( new int[]{d,k}, value );
                }
                if( col>=0 ) {
                    for (int m = 0; m < 5; m++) {
                        vector1.putScalar(new int[]{d, col + m * 24}, 0);
                      /*  vector1.putScalar(new int[]{d, 1 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 3 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 5 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 6 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 7 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 9 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 10 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 17 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 19 + m * 24}, 0);
                        vector1.putScalar(new int[]{d, 22 + m * 24}, 0);*/
                    }
                }
            }
/*

0 closeMA200,closeMA200xo,closeMA50,closeMA50xo,"+
4 cmf,macd,macdSignal,atrDaily,atr,atrPct,
10 mfi,pvo,obv,willR,"+
14 kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct,"+
20 macdv,macdvSignal,mPhase,mDir

f1(0)=0.4138 - closeMA200 **
f1(14)=0.5379 - kcLwr **
f1(16)=0.5467 - KcUpr **
f1(1)=0.8927 - closeMA200xo
f1(2)=0.8417 - closeMA50,
f1(23)=0.7578 - mDir
f1(18)=0.7270 - kcMPct **
f1(15)=0.7795 - kcMid
f1(11)=0.7975 - pvo
f1(20)=0.7974 - macdv
f1(4)=0.8267 - cmf
f1(21)=0.8322 - macdvSignal
f1(13)=0.8488 - willR
f1(12)=0.8555 - obv
f1(22)=0.8906 - mPhase

f1(3)=0.9477 -
f1(5)=1.0000
f1(6)=1.0000
f1(7)=0.9608
f1(8)=0.9498
f1(9)=0.9920
f1(10)=0.9648
f1(17)=0.9726
f1(19)=0.9272

 */
            INDArray output1 = net.output( vector1 );
            Evaluation eval3a = new Evaluation();
            eval3a.eval(ds.getLabels(), output1);
            //System.out.println(eval3a.stats());
            double f1 = eval3a.f1();
            System.out.format("f1(%d)=%.4f, b=[%.2f, %.2f], s=[%.2f, %.2f]\n", col, f1, eval3a.precision(1),eval3a.recall(1), eval3a.precision(2),eval3a.recall(2));
        }


    }

    double[] price;
    double[] buy;
    double[] sell;
    double[][] inputs;
    int numDays;
    int VECTOR_SIZE;

    public Forecast(int MAX_SIZE, int _VECTOR_SIZE) {
        VECTOR_SIZE = _VECTOR_SIZE;
        price = new double[MAX_SIZE];
        buy = new double[MAX_SIZE];
        sell = new double[MAX_SIZE];
        inputs = new double[MAX_SIZE][VECTOR_SIZE];
        numDays = 0;
    }
    private DataSet loadDataSet(String tsFile, int fromYear, int toYear) {
        try (BufferedReader br = new BufferedReader(new FileReader(tsFile))) {
            String line;
            numDays = 0;
            while ((line = br.readLine()) != null) {
                String[] fields = line.split(",");
                if (fields.length < (VECTOR_SIZE+5) ) continue;

                int year = Integer.parseInt(fields[0].substring(0, 4));

                if (year >= fromYear && year <= toYear ) {
                    price[numDays] = Double.parseDouble(fields[1]);
                    buy[numDays] = Double.parseDouble(fields[3]);
                    sell[numDays] = Double.parseDouble(fields[4]);
                    for (int i = 0; i < VECTOR_SIZE; i++) {
                        inputs[numDays][i] = Double.parseDouble(fields[5 + i]);
                    }
                    numDays++;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        INDArray features = Nd4j.create(numDays, VECTOR_SIZE);
        INDArray labels = Nd4j.create(numDays, 3);

        //System.out.println(Arrays.toString(features.shape()));
        //System.out.println(Arrays.toString(labels.shape()));

        for (int i = 0; i < numDays; i++) {
            features.putRow(i, Nd4j.create(inputs[i]));
            double[] y = new double[3];
            y[0] = 1;
            y[1] = 0;
            y[2] = 0;
            if (buy[i] > 0.5) {
                y[0] = 0;
                y[1] = 1;
            }
            if (sell[i] > 0.5) {
                y[0] = 0;
                y[2] = 1;
            }
            labels.putRow(i, Nd4j.create(y));
        }
        DataSet ds = new DataSet(features, labels);
        //System.out.println(ds);
        return ds;
    }


    public DataSet loadDataSet(BacktestingProcessor bp) {

        numDays = bp.totalDays;

        INDArray features = Nd4j.create(numDays, VECTOR_SIZE);
        INDArray labels = Nd4j.create(numDays, 3);

        //System.out.println(Arrays.toString(features.shape()));
        //System.out.println(Arrays.toString(labels.shape()));

        for (int i = 0; i < numDays; i++) {
            double[] _inputs = bp.stockData.inputVector[i];
            features.putRow(i, Nd4j.create(_inputs));
            double[] y = new double[3];
            y[0] = 1;
            y[1] = 0;
            y[2] = 0;
            if (bp.stockData.buySignal[i] > 0.5) {
                y[0] = 0;
                y[1] = 1;
            }
            if (bp.stockData.sellSignal[i] > 0.5) {
                y[0] = 0;
                y[2] = 1;
            }
            labels.putRow(i, Nd4j.create(y));
        }
        DataSet ds = new DataSet(features, labels);
        //System.out.println(ds);
        return ds;
    }
}
