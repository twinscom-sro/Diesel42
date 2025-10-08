package vectorTasks;

import datamodels.TradeData;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class IndicatorsSet {

    String tickerId;
    int dataSize;

    //tradedata vectors
    public String[] dates;
    public double[] open;
    public double[] high;
    public double[] low ;
    public double[] close;
    public double[] adjOpen;
    public double[] adjHigh;
    public double[] adjLow ;
    public double[] adjClose;
    public double[] volume;
    double[] dividend;
    double[] split;

    //SIGNAL data
    Series pf8;
    Series zigZag8;
    Series pf15;
    Series buySignal8;
    Series sellSignal8;
    Series buySignal15;
    Series sellSignal15;
    
    //KPI indicators
    Series closeMA200;
    public Series closeMA200xo;
    Series closeMA50;
    public Series closeMA50xo;
    public Series cmf;
    public Series macd;
    public Series macdSignal;
    Series atrDaily;
    Series atrDailyPct;
    Series atr;
    public Series atrPct;
    public Series mfi;
    public Series pvo;
    public Series obv;
    public Series willR;
    Series kcMid;
    Series kcUpper;
    Series kcLower;
    public Series kcUpperATR;
    public Series kcMidATR;
    public Series kcLowerATR;
    public Series moonPhase;
    public Series moonDirection;

    public Series ema12;
    public Series ema26;
    public Series atr26;
    public Series macdv;
    public Series macdvSignal;


    public IndicatorsSet(String ticker) {
        tickerId = ticker;
    }

    public void recalculateOHLC(List<TradeData> dataSet) {

        dataSize = dataSet.size();
        
        dates = new String[dataSize];
        open = new double[dataSize];
        high = new double[dataSize];
        low = new double[dataSize];
        close = new double[dataSize];
        volume = new double[dataSize];
        dividend = new double[dataSize];
        split = new double[dataSize];
        adjOpen = new double[dataSize];
        adjHigh = new double[dataSize];
        adjLow = new double[dataSize];
        adjClose = new double[dataSize];
        double[] reinvestment = new double[dataSize];
        double[] qty = new double[dataSize];
        double[] value = new double[dataSize];

        int i=dataSize-1;
        // data is served in descending order (latest first)
        for( TradeData s : dataSet ) {
            if( i < 0 ) break;
            dates[i] = s.date;
            open[i] = s.open;
            close[i] = s.close;
            high[i] = s.high;
            low[i] = s.low;
            volume[i] = s.volume;
            dividend[i] = s.dividend;
            split[i] = s.splitCoef;
            adjOpen[i] = s.open;
            adjClose[i] = s.close;
            adjHigh[i] = s.high;
            adjLow[i] = s.low;
            reinvestment[i] = dividend[i]/close[i];
            i--;
        }

        qty[0]=1;
        value[0]=close[0];
        for( int j=1; j<dataSize; j++ ){
            qty[j] =  (qty[j-1]+reinvestment[j-1])*split[j];
            value[j] = qty[j]*close[j];
        }

        for( int j=dataSize-2; j>=0; j-- ){
            double recalculatedClose = value[j]/value[j+1]*adjClose[j+1];
            adjClose[j] = recalculatedClose;
            adjOpen[j] = open[j]/close[j]*recalculatedClose;
            adjHigh[j] = high[j]/close[j]*recalculatedClose;
            adjLow[j] =  low[j]/close[j]*recalculatedClose;
        }        
    }

    public void recalculateStandardKPIs() {
        closeMA200 = Indicators.calculate_ma( 200, new Series(adjClose) );
        closeMA200xo = Indicators.calculate_xo_pct( new Series(adjClose), closeMA200 );
        closeMA50 = Indicators.calculate_ma( 50, new Series(adjClose) );
        closeMA50xo = Indicators.calculate_xo_pct( new Series(adjClose), closeMA50 );
        cmf = Indicators.calculate_cmf( 10, new Series(adjClose), new Series(adjHigh), new Series(adjLow), new Series(volume) );
        macd = Indicators.calculate_macd( new Series(adjClose) );
        macdSignal = Indicators.calculate_ema( 9, macd );
        atrDaily = Indicators.calculate_atr( new Series(adjHigh), new Series(adjLow), new Series(adjClose) );
        atr = Indicators.calculate_ema( 11, atrDaily );
        atrDailyPct = Indicators.calculate_pct( atrDaily, new Series(adjClose) );
        atrPct = Indicators.calculate_ema( 11, atrDailyPct );
        mfi = Indicators.calculate_mfi( 14, new Series(adjHigh), new Series(adjLow), new Series(adjClose), new Series(volume) );
        pvo = Indicators.calculate_pvo( new Series(volume)  );
        obv = Indicators.calculate_obv( new Series(adjClose), new Series(volume)  );
        willR = Indicators.calculate_willR( new Series(adjHigh), new Series(adjLow), new Series(adjClose)  );

        ema12 = Indicators.calculate_ema( 12, new Series(adjClose) );
        ema26 = Indicators.calculate_ema( 26, new Series(adjClose) );
        atr26 = Indicators.calculate_ema( 26, atrDaily );
        macdv = new Series(dataSize);
        for( int d=0; d<atr26.length(); d++ ){
            macdv.x[d] = (ema12.x[d]-ema26.x[d])/atr26.x[d]*100;
        }
        macdvSignal = Indicators.calculate_ema( 9, macdv );

        kcMid = Indicators.calculate_ema( 20, new Series(adjClose) );
        kcUpper = kcMid.addMult(atr, 2);
        kcLower = kcMid.addMult(atr, -2);

        kcUpperATR = Indicators.calculate_xo_pct(kcUpper,new Series(adjClose));
        kcLowerATR = Indicators.calculate_xo_pct(kcLower,new Series(adjClose));
        kcMidATR = Indicators.calculate_xo_ratio(new Series(adjClose),kcMid,atr);

        pf8 = new Series(adjClose).calculatePF( 8.0 );
        zigZag8 = pf8.calculateZigZag();
        pf15 = new Series(adjClose).calculatePF( 15.0 );
        buySignal8 = pf8.generateBuySignal(1, 2, 2);
        sellSignal8 = pf8.generateSellSignal(1, 3, 1);
        adjustSignals(buySignal8.x,sellSignal8.x);

        buySignal15 = pf15.generateBuySignal(1, 2, 2);
        sellSignal15 = pf15.generateSellSignal(1, 3, 1);
        adjustSignals(buySignal15.x,sellSignal15.x);

        moonPhase = new Series(dataSize);
        moonDirection = new Series(dataSize);
        Calendar c = Calendar.getInstance();
        MoonPhase mp = new MoonPhase(c);

        for( int d=0; d<dataSize; d++ ) {
            int yyyy = Integer.parseInt(dates[d].substring(0, 4));
            int mm = Integer.parseInt(dates[d].substring(5, 7));
            int dd = Integer.parseInt(dates[d].substring(8, 10));
            //c.set( yyyy, mm-1, dd );
            //mp.updateCal(c);
            float mpPos = (float) mp.getPhase(yyyy, mm, dd);  // CHECK CONSISTENCY OF CALCULATIONS!!!
            moonPhase.set( d, (float) (Math.abs(mpPos) ) );
            moonDirection.set( d, (float) (mpPos > 0 ? 100.0 : (mpPos < 0 ? -100.0 : 0.0)) );
        }
    }

    private void adjustSignals(float[] a, float[] b) {
        List<Integer> mixedSignals = new ArrayList<Integer>();
        for( int i=0; i<a.length; i++ ) {
            float buy = a[i] + (i>0 ? a[i-1] : 0) + (i<a.length-1 ? a[i+1] : 0);
            float sell = b[i] + (i>0 ? b[i-1] : 0) + (i<b.length-1 ? b[i+1] : 0);
            if( a[i]>0.5 && b[i]>0.5 ){
                mixedSignals.add(i);
            }
        }
        for( int i : mixedSignals ) {
            a[i]=0;
            b[i]=0;
        }
    }


    public void writeDataToKPI(String fileName) {
        File csvOutputFile = new File(fileName);
        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
            pw.println("dates,open,high,low,close,volume,dividend,split,adjOpen,adjHigh,adjLow,adjClose,"+
                    "closeMA200,closeMA200xo,closeMA50,closeMA50xo," +
                    "cmf,macd,macdSignal,atrDaily,atr,atrPct,mfi,pvo,obv,willR," +
                    "kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct,macdv,macdvSignal,mPhase,mDir,"+
                    "pf8,zigZag,pf15,buySignal8,sellSignal8,buySignal15,sellSignal15");
            for( int i=0; i<dataSize; i++ ) {
                String data = String.format(
                        "%10s, %6.2f, %6.2f, %6.2f, %6.2f, %10.0f, %6.2f, %6.4f, %6.2f, %6.2f, %6.2f, %6.2f",
                        dates[i], open[i],high[i],low[i],close[i],volume[i],
                        dividend[i],split[i],
                        adjOpen[i],adjHigh[i],adjLow[i],adjClose[i] );
                String kpi = String.format(
                        "%6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f"+
                                ", %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f"+
                                ", %6.2f, %6.2f"+
                                ", %6.2f, %6.2f"+
                                ", %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f, %6.5f",
                        closeMA200.x[i], closeMA200xo.x[i], closeMA50.x[i], closeMA50xo.x[i],
                        cmf.x[i], macd.x[i], macdSignal.x[i], atrDaily.x[i], atr.x[i], atrPct.x[i], mfi.x[i], pvo.x[i], obv.x[i], willR.x[i],
                        kcLower.x[i],kcMid.x[i],kcUpper.x[i],kcLowerATR.x[i],kcMidATR.x[i],kcUpperATR.x[i],
                        macdv.x[i],macdvSignal.x[i],
                        moonPhase.x[i], moonDirection.x[i],
                        pf8.x[i], zigZag8.x[i], pf15.x[i], buySignal8.x[i], sellSignal8.x[i], buySignal15.x[i], sellSignal15.x[i] );
                pw.println(String.format("%s, %s",data,kpi));
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
