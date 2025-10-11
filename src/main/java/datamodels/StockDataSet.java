package datamodels;

import environment.Utilities;
import neural.DeepLayer;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class StockDataSet {

    public String[] dates;
    public double[][] inputVector;
    public String[] inputKPIs;
    public double[] buySignal;
    public double[] sellSignal;
    public double[] zigZag;
    public double[] price;
    public int inputSize;
    public int samples;

    /*
    All KPI columns:
    dates,open,high,low,close,volume,dividend,split,adjOpen,adjHigh,adjLow,adjClose,
    closeMA200,closeMA200xo,closeMA50,closeMA50xo,
    cmf,macd,macdSignal,atrDaily,atr,atrPct,mfi,pvo,obv,willR,
    kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct,
    macdv,macdvSignal,mPhase,mDir,
    pf8,zigZag,pf15,buySignal8,sellSignal8,buySignal15,sellSignal15
     */
    public void loadDataSet(String dataFile, String[] filters, String[] _inputKPIs, int history) {
        inputKPIs = _inputKPIs;
        List<String> days = new ArrayList<>();
        List<Integer> inputColumns = new ArrayList<>();
        List<double[]> records = new ArrayList<>();
        List<Integer> priceColumns = new ArrayList<>();
        List<double[]> pricing = new ArrayList<>();
        String[] priceVector = {"buySignal8","sellSignal8","pf8","zigZag","close","adjClose"};
        String line;

        // Use try-with-resources to ensure the BufferedReader is closed automatically
        try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {

            int nLine=0;
            while ((line = br.readLine()) != null) {
                if( nLine++==0 ){
                    String[] values = line.split(","); //DEFAULT_DELIMITER + "(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                    for( String kpi : inputKPIs ){
                        //System.out.println("Scanning columns for "+kpi);
                        int col=0;
                        for( String v : values ) {
                            if (kpi.contentEquals(v)) {
                                inputColumns.add(col);
                            }
                            col++;
                        }
                    }
                    //System.out.println( "Input columns vector = "+Arrays.toString(inputColumns.toArray()) );
                    for( String y : priceVector ){
                        int col=0;
                        for( String v : values ) {
                            if (y.contentEquals(v)) {
                                priceColumns.add(col);
                            }
                            col++;
                        }
                    }
                    inputSize = inputColumns.size();
                    //System.out.println( "Output columns vector = "+Arrays.toString(priceColumns.toArray()) );
                    if(inputSize==0) return;
                    continue;
                }
                String[] values = line.split(","); //DEFAULT_DELIMITER + "(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                String year = values[0].substring(0, 4);
                boolean includeFlag=false;
                for( String yearFilter : filters ){
                    if( yearFilter.equals(year) ){
                        includeFlag=true;
                        break;
                    }
                }
                if( includeFlag ){
                    // Clean up any residual quotes from the split process
                    days.add( values[0] );
                    int pos = 0;
                    double[] v = new double[inputSize];
                    for (int i : inputColumns ) {
                        v[pos++] = Double.parseDouble( values[i] );
                    }
                    records.add(v);
                    double[] y = new double[priceColumns.size()];
                    pos=0;
                    for (int i : priceColumns ) {
                        y[pos++] = Double.parseDouble( values[i] );
                    }
                    pricing.add(y);

                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //System.out.println("records=" + records.size());
        //System.out.println("cols=" + records.get(0).length);

        samples = records.size();
        if( history<1 )history=1;
        if( history>10 )history=10;
        inputVector = new double[samples][inputSize*history];
        dates = new String[samples];
        buySignal = new double[samples];
        sellSignal = new double[samples];
        zigZag = new double[samples];
        price = new double[samples];

        for( int k=0; k<samples; k++ ) {
            dates[k] = days.get(k);
            for( int i=0; i < inputSize; i++) {
                inputVector[k][i] = records.get(k)[i];
            }
            buySignal[k] = pricing.get(k)[0];
            sellSignal[k] = pricing.get(k)[1];
            zigZag[k] = pricing.get(k)[3];
            price[k] = pricing.get(k)[5];

            //if( k%50==0) System.out.println( k + "["+dates[k]+"]-> " + Arrays.toString(inputVector[k]) );
        }
        normalizeInputs("closeMA200xo",4.74292099,14.26543601);
        normalizeInputs("closeMA50xo",1.27341681,6.92262897);
        normalizeInputs("cmf",0.03265432,0.20682017);
        normalizeInputs("macd",0.52668275,3.23695993);
        normalizeInputs("macdSignal",0.51746509,3.04182862);
        normalizeInputs("atrDaily",3.88681522,6.09221636);
        normalizeInputs("atr",3.88247562,5.75621837);
        normalizeInputs("atrPct",2.3035259,3.42048145);
        normalizeInputs("mfi",52.80884259,76.37326945);
        normalizeInputs("pvo",-0.69947629,8.27405592);
        normalizeInputs("obv",3.74872051,24.64325854);
        normalizeInputs("willR",44.20454974,69.46943854);
        normalizeInputs("kcLPct",-4.90150646,7.7722257);
        normalizeInputs("kcMPct",0.30185095,1.48703185);
        normalizeInputs("kcUPct",4.29372565,7.92920582);
        normalizeInputs("macdv",19.9256551,82.23265115);
        normalizeInputs("macdvSignal",19.73303331,77.78983534);
        normalizeInputs("mPhase",49.90514991,78.88413791);
        normalizeInputs("mDir",0.52910053,100.03080758);

        for( int k=0; k<samples; k++ ) {
            for( int i=0; i < inputSize; i++) {
                for (int h = 0; h < history; h++) {
                    inputVector[k][i + h * inputSize] = (k-h) >= 0 ? inputVector[k-h][i] : 0;
                }
            }
        }
        // finally, extend the input vector
        inputSize = inputSize*history;
    }

    public void normalizeInputs(String kpi, double mu, double sigma) {
        for( int i=0; i< inputSize; i++ ){
            if( inputKPIs[i].contentEquals(kpi) ) {
                for( int d = 0; d < inputVector.length; d++) {
                    inputVector[d][i] = (inputVector[d][i]-mu)/sigma;
                }
                break;
            }
        }
    }

}
