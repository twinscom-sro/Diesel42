package twc;

import datamodels.JSON;
import environment.Utilities;
import neural.DeepLayer;
import optimizationTasks.ModelMixer;
import org.bson.Document;
import predictorTasks.BacktestingProcessor;
import trainingTasks.TrainingProcessor;
import vectorTasks.VectorsProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    /*public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        System.out.printf("Hello and welcome!");

        for (int i = 1; i <= 5; i++) {
            //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
            // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
            System.out.println("i = " + i);
        }
    }*/

    static final String MENU = """
    Main menu:
    0 - display help
    1 - load/update KPI files. usage: java -jar Diesel.jar 1 <kpi> <netw> <out> "tkr1,tkr2...."
    2 - train buy/sell networks. usage: java -jar Diesel.jar 2 <kpi> <netw> <out> "tkr1,tkr2...." <period-filter> <config> <mult> <iterations> <neurons>
    3 - backtest buy/sell networks. usage: java -jar Diesel.jar 3 <kpi> <netw> <out> <model> "tkr1,tkr2...." <period-filter> <config> <mult> 
    4 - optimize models. usage: java -jar Diesel.jar 4 <kpi> <netw> <out> <tkr> "model1,model2,...." <period-filter> <config> <mult> 
    5 - optimize models. usage: java -jar Diesel.jar 5 <kpi> <netw-set> <out> <tkr> <model> <period-filter> <config> <mult>, optimize based on a directory of models
    6 - backtest models. usage: java -jar Diesel.jar 6 <2=symbol> <3=buyModels> <4=sellModels> <5=buyThreshold> <6=sellThreshold> <7=config> <8=mult> <9=neurons>
    7 - backtest from model folder. usage: java -jar Diesel.jar 7 <kpi> <netw> <out> <tkr> <period-filter> <config> <multi> 
    8 - optimize models from folder. usage: java -jar Diesel.jar 5 <kpi> <netw-set> <out> <tkr> <period-filter> <config> <mult>, optimize based on a directory of models
    """;

    //debugging:
    static final String ITERATIONS = "10000";
    static final String NEURONS = "256"; //"1024";
    static final String KPIS = "c:/_db/kpis";
    static final String NETS = "c:/_db/nets64_vega1";
    static final String OUTS = "c:/_arcturus/2025-10-13a";
    static final String VECTOR = "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal";
    static final String HISTORY = "3";


    static final String NEURONS2 = "4096"; //"1024";
    static final String NETS2 = "c:/_db/nets_120";
    static final String OUTS2 = "c:/_arcturus/2025-10-22a";
    static final String VECTOR2 = "closeMA200,closeMA200xo,closeMA50,closeMA50xo,cmf,"+
            "macd,macdSignal,atrDaily,atr,atrPct,mfi,pvo,obv,willR,kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct," +
            "macdv,macdvSignal,mPhase,mDir";
    static final String HISTORY2 = "5";


    static final String[] option1 = { "1", KPIS, NETS, OUTS,
              "WMT,GS,MSFT,CAT,HD,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL,CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ,MSTR" };

    /*
    {   methodId: "UPDATE_KPI",
        tickerId: "IBM",
        kpiDB: "~/db/kpis"
    }
     */
    static final String[] option2 = { "2", KPIS, NETS, OUTS,
            "WMT,GS,MSFT",
            //"CAT,HD,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL",
            //"CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO",
            //"NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ,MSTR"
            "2016,2017,2018,2019,2022,2023",
            VECTOR, HISTORY, ITERATIONS, NEURONS};
    static final String[] option2b = { "2b", KPIS, NETS, OUTS,
            "WMT,GS,MSFT",
            "1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
            VECTOR, HISTORY, ITERATIONS, NEURONS};
    /*
    {   methodId: "TRAIN_MODEL5",
        tickerId: "IBM",
        kpiDB: "~/db/kpis",
        modelDB: "~/db/nets",
        periods: [2016,2017,2018,2019,2022,2023],
        trainVector: ["cmf","obv","willR","atrPct","kcMPct","kcUPct","macdv","macdvSignal"],
        historyMultiplier: 3,
        iterations: 100000,
        neurons: 64
    }
     */


    static final String[] option3 = { "3", KPIS, NETS, OUTS,
            "CAT,WMT,GS,MSFT,CAT,HD,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL,CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ",
            "WMT,GS,MSFT,CAT,HD,MCD,CRM,BA,JNJ,PG,DIS,MRK,NKE,KO,VZ",
            "2023,2024,2025",
            VECTOR, HISTORY, NEURONS};
    static final String[] option4 = { "4", KPIS, NETS, OUTS,
            "CAT",
            "WMT,GS,MSFT,CAT,HD,MCD,CRM,BA,JNJ,PG,DIS,MRK,NKE,KO,VZ",
            "2021,2022,2023,2024",
            VECTOR, HISTORY, NEURONS};
    static final String[] option5 = { "5", KPIS, "C:/_db/nets256_vega3/", "C:/_arcturus/2025-10-15-vega3/",
            "X","2021,2022,2023,2024", VECTOR, HISTORY, NEURONS};
    /*
    {   methodId: "OPTIMIZE5",
        tickerId: "IBM",
        kpiDB: "~/db/kpis",
        modelDB: "~/db/nets",
        periods: [2022,2023,2023],
        trainVector: ["cmf","obv","willR","atrPct","kcMPct","kcUPct","macdv","macdvSignal"],
        historyMultiplier: 3,
        iterations: 100000,
        neurons: 64,
        forecast: [2024,2025],
    }
     */
    static String[] CVX_buyComponents = {
            "C:/_db/nets64_vega1/CAT_network2.txt",
            "C:/_db/nets64_vega1/HD_network1.txt",
            "C:/_db/nets64_vega1/JPM_network1.txt",
            "C:/_db/nets64_vega1/KO_network2.txt",
            "C:/_db/nets64_vega1/MCD_network2.txt",
            "C:/_db/nets64_vega1/MMM_network2.txt",
            "C:/_db/nets64_vega1/MPW_network1.txt",
            "C:/_db/nets64_vega1/NKE_network1.txt",
            "C:/_db/nets64_vega1/PLUG_network2.txt",
            "C:/_db/nets64_vega1/SHW_network1.txt",
            "C:/_db/nets64_vega1/WMT_network2.txt" };
    static String[] CVX_sellComponents = {
            "C:/_db/nets64_vega1/CRM_network2.txt",
            "C:/_db/nets64_vega1/PLUG_network1.txt",
            "C:/_db/nets64_vega1/SHW_network2.txt" };
    //buyThreshold=3
    //sellThreshold=1

    static final String[] option6 = {"6", "CVX", KPIS+"\\", OUTS+"\\", String.join(",",CVX_buyComponents), String.join(",",CVX_sellComponents), "3", "1",
            "2024,2025", VECTOR, HISTORY, NEURONS };

    static final String[] option7 = {"7", KPIS, NETS2, OUTS2, "2024,2025", VECTOR2, HISTORY2, NEURONS2 };
    static final String[] option8 = {"8", KPIS, NETS2, OUTS2, "2019,2020,2021,2022,2023", VECTOR2, HISTORY2, NEURONS2 };


    /*
    {   methodId: "FORECAST5",
        tickerId: "IBM",
        kpiDB: "~/db/kpis",
        modelDB: "~/db/nets",
        forecast: [2024,2025],
        trainVector: ["cmf","obv","willR","atrPct","kcMPct","kcUPct","macdv","macdvSignal"],
        historyMultiplier: 3,
        iterations: 100000,
        neurons: 64,
        buyModels: [  ... files ... ],
        sellModels: [  ... files ... ],
        buyThreshold: 3,
        sellThreshold: 1
    }
     */

    /*
    LINUX 10/11/2025:
java -jar Diesel42.2v.jar 1 ./kpis1011 ./nets1011a ./out1011a "WMT,GS,MSFT,CAT,HD,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL,CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ"
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011a ./out1011a "WMT" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 50000 4096 >t1.txt &
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011a ./out1011a "WMT,GS,MSFT,CAT,HD" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 50000 >t1.txt &

,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL,CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ"


run nets1011b:
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011b ./out1011b "WMT,GS,MSFT,CAT,HD" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 200000 603 > u1.txt &
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011b ./out1011b "UNH,V,SHW,AXP,JPM" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 200000 603 > u2.txt &
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011b ./out1011b "MCD,AMGN,IBM,TRV,AAPL" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 200000 603 > u3.txt &
java -jar Diesel42.2v.jar 2 ./kpis1011 ./nets1011b ./out1011b "CRM,BA,AMZN,HON,JNJ" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 200000 603 > u4.txt &
NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ

UBUNTU 10/12/25:
java -jar ./diesel/Diesel42.2.jar 1 ./kpis ./nets ./out1 "WMT,GS,MSFT,CAT,HD,UNH,V,SHW,AXP,JPM,MCD,AMGN,IBM,TRV,AAPL,CRM,BA,AMZN,HON,JNJ,NVDA,MMM,CVX,PG,DIS,MRK,CSCO,NKE,KO,VZ,COIN,SIL,MPW,PLUG,NNBR,GDXJ,MSTR,SPX"

java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "WMT,GS,MSFT,CAT,HD" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u1.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "UNH,V,SHW,AXP,JPM" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u2.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "MCD,AMGN,IBM,TRV,AAPL" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u3.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "CRM,BA,AMZN,HON,JNJ" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u4.txt &

java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "NVDA,MMM,CVX,PG,DIS" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u5.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "MRK,CSCO,NKE,KO,VZ" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u6.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "COIN,SIL,MPW,PLUG,NNBR" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u7.txt &
java -jar ./diesel/Diesel42.2.jar 2 ./kpis ./nets ./out1 "GDXJ,MSTR" "2016,2017,2018,2019,2021,2022,2023" "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal" 3 100000 603 > u8.txt &

,,

     */

    public static void main(String[] args) {

        //debugging:
        args = option8;
        String MDB = "C:/_db/models/optimizer.txt";

        System.out.println("Running option: "+ Arrays.toString(args));

        if( args.length < 5 ) {
            System.out.println(MENU);
            return;
        }
        String task=args[0];
        String KPI= args[1]+"/";
        String NET= args[2]+"/";
        String OUT= args[3]+"/";
        String GROUP= args[4];
        String[] tkr = GROUP.split(",");
        if( task.contentEquals("1") ){
            for( String t : tkr ) loadKPI( t, KPI, OUT );
        }else if( task.contentEquals("2") ){
            String[] periods = args[5].split(",");
            String[] config = args[6].split(",");
            int multiplier = Integer.parseInt(args[7]);
            int iterations = Integer.parseInt(args[8]);
            int neurons = Integer.parseInt(args[9]);
            for( String t : tkr ) trainTicker( t, KPI, NET, OUT, periods, config, multiplier, iterations, neurons );
        }else if( task.contentEquals("2b") ){
            String[] periods = args[5].split(",");
            String[] config = args[6].split(",");
            int multiplier = Integer.parseInt(args[7]);
            int iterations = Integer.parseInt(args[8]);
            int neurons = Integer.parseInt(args[9]);
            for( String t : dji30c ) trainingSet( t, KPI, NET, OUT, periods, config, multiplier, iterations, neurons );
        }else if( task.contentEquals("3") ){
            String[] tkrSet = args[4].split(",");
            String[] models = args[5].split(",");
            String[] periods = args[6].split(",");
            String[] config = args[7].split(",");
            int neurons = Integer.parseInt(args[9]);
            for( String tkr1 : tkrSet ) backtestTicker( tkr1, KPI, NET, OUT, models, periods, config, Integer.parseInt(args[8]), neurons );
        }else if( task.contentEquals("4") ){
            String tkrRef = args[4];
            String[] models = args[5].split(",");
            String[] periods = args[6].split(",");
            String[] config = args[7].split(",");
            int neurons = Integer.parseInt(args[9]);
            /*for( String t : dji30 )*/ optimizeModels( tkrRef, KPI, NET, OUT, models, periods, config, Integer.parseInt(args[8]), neurons );
        }else if( task.contentEquals("5") ){
            String modelFolder = args[2];
            String tickerId = args[4];
            String[] periods = args[5].split(",");
            String[] config = args[6].split(",");
            int history =  Integer.parseInt(args[7]);
            int neurons = Integer.parseInt(args[8]);
            StringBuilder mdb = new StringBuilder();
            for( String t : dji30c ) optimizeModelsFromFolder( mdb, t, KPI, OUT, modelFolder, periods, config, history, neurons );
            //optimizeModelsFromFolder( mdb,"PLUG", KPI, OUT, modelFolder, periods, config, history, neurons );
            //optimizeModelsFromFolder( mdb,"SIL", KPI, OUT, modelFolder, periods, config, history, neurons );
            Utilities.writeFile(MDB,mdb);
        }else if( task.contentEquals("6") ){
//            6 - backtest models. usage: java -jar Diesel.jar 6 <2=symbol> <3=buyModels> <4=sellModels> <5=buyThreshold> <6=sellThreshold> <7=config> <8=mult> <9=neurons>
                    String tickerId = args[1];
                    String KPI1 = args[2];
                    String OUT1 = args[3];
                    String[] buyModels = args[4].split(",");
                    String[] sellModels = args[5].split(",");
                    int buyThreshold = Integer.parseInt(args[6]);
                    int sellThreshold = Integer.parseInt(args[7]);
                    String[] periods = args[8].split(",");
                    String[] config = args[9].split(",");
                    int multiplier = Integer.parseInt(args[10]);
                    int neurons = Integer.parseInt(args[11]);
                    backtestModels( tickerId, KPI1, OUT1, buyModels, sellModels,buyThreshold, sellThreshold, periods, config, multiplier, neurons );
        }else if( task.contentEquals("7") ){
            //"7", KPIS+"\\", NETS2+"\\", OUTS2+"\\", "2024,2025", VECTOR2, HISTORY2, NEURONS2
            String[] periods = args[4].split(",");
            String[] config = args[5].split(",");
            int multiplier = Integer.parseInt(args[6]);
            int neurons = Integer.parseInt(args[7]);
            for( String t : dji30c ){
                backtestTicker2( t, KPI, NET, OUT, periods, config, multiplier, neurons );
                //break;
            }
        }else if( task.contentEquals("8") ){
            String[] periods = args[4].split(",");
            String[] config = args[5].split(",");
            int multiplier = Integer.parseInt(args[6]);
            int neurons = Integer.parseInt(args[7]);
            StringBuilder mdb = new StringBuilder();
            for( String t : dji30c ){
                optimizeModelsFromFolder2( mdb, t, KPI, NET, OUT, periods, config, multiplier, neurons );
                //break;
            }
        }else{
            System.out.println( MENU );
        }
    }





/*
"cmf,obv,willR,kcMPct,kcUPct,macdv,macdvSignal"
"WMT,GS,CAT,CRM,JNJ,KO"
 */

    static String[] dji30 = {"WMT","GS","MSFT","CAT","HD","UNH","V","SHW","AXP","JPM",
    "MCD","AMGN","IBM","TRV","AAPL","CRM","BA","AMZN","HON","JNJ","NVDA","MMM",
    "CVX","PG","DIS","MRK","CSCO","NKE","KO","VZ"};
    static String[] dji30a = {"WMT","GS","CAT","CRM","JNJ","KO"};
    static String[] dji30b = {"WMT","GS","CRM","JNJ"};

    static String[] dji30c = {"WMT","GS","MSFT","CAT","HD","UNH","V","SHW","AXP","JPM",
            "MCD","AMGN","IBM","TRV","AAPL","CRM","BA","AMZN","HON","JNJ","NVDA","MMM",
            "CVX","PG","DIS","MRK","CSCO","NKE","KO","VZ","COIN","NNBR","MPW", "MSTR", "GDXJ", "NVDA", "PLUG"/*, "SIL"*/ };

    public static void loadKPI( String tkr, String KPI, String OUT ){
        VectorsProcessor vp = new VectorsProcessor();
        String fileName = KPI+tkr+"_kpis.txt";
        JSON json = vp.runTask(tkr, fileName);
    }

    public static void main_02(String[] args) {
        // get scaling factors
        TrainingProcessor tp = new TrainingProcessor();
        String[] filters = {"2019","2020","2021","2022","2023","2024","2025"};
        //String[] params ={"cmf","macd","macdSignal","obv","macdv","macdvSignal"};
        String[] params ={"closeMA200","closeMA200xo","closeMA50","closeMA50xo",
                "cmf","macd","macdSignal","atrDaily","atr","atrPct",
                "mfi","pvo","obv","willR",
                "kcLwr","kcMid","KcUpr","kcLPct","kcMPct","kcUPct",
                "macdv","macdvSignal","mPhase","mDir"};
        double[][] avg = new double[dji30.length][params.length];
        double[][] stdev = new double[dji30.length][params.length];
        int item=0;
        for( String tkr : dji30 ) {
            String kpiFile = "c:/_db/"+tkr+"_kpis.txt";

            tp.loadDataSet( kpiFile, filters, params, 1);
            tp.normalizeInputs("closeMA200xo",4.74292099,14.26543601);
            tp.normalizeInputs("closeMA50xo",1.27341681,6.92262897);
            tp.normalizeInputs("cmf",0.03265432,0.20682017);
            tp.normalizeInputs("macd",0.52668275,3.23695993);
            tp.normalizeInputs("macdSignal",0.51746509,3.04182862);
            tp.normalizeInputs("atrDaily",3.88681522,6.09221636);
            tp.normalizeInputs("atr",3.88247562,5.75621837);
            tp.normalizeInputs("atrPct",2.3035259,3.42048145);
            tp.normalizeInputs("mfi",52.80884259,76.37326945);
            tp.normalizeInputs("pvo",-0.69947629,8.27405592);
            tp.normalizeInputs("obv",3.74872051,24.64325854);
            tp.normalizeInputs("willR",44.20454974,69.46943854);
            tp.normalizeInputs("kcLPct",-4.90150646,7.7722257);
            tp.normalizeInputs("kcMPct",0.30185095,1.48703185);
            tp.normalizeInputs("kcUPct",4.29372565,7.92920582);
            tp.normalizeInputs("macdv",19.9256551,82.23265115);
            tp.normalizeInputs("macdvSignal",19.73303331,77.78983534);
            tp.normalizeInputs("mPhase",49.90514991,78.88413791);
            tp.normalizeInputs("mDir",0.52910053,100.03080758);
            /*
dates,open,high,low,close,volume,dividend,split,adjOpen,adjHigh,adjLow,adjClose,
closeMA200,closeMA200xo,closeMA50,closeMA50xo,
cmf,macd,macdSignal,atrDaily,atr,atrPct,
mfi,pvo,obv,willR,
kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct,
macdv,macdvSignal,mPhase,mDir,
pf8,zigZag,pf15,buySignal8,sellSignal8,buySignal15,sellSignal15




             */
            System.out.println(kpiFile);
            tp.calculateKPIStats();
            for( int i=0; i<params.length; i++ ) {
                avg[item][i] = tp.avg[i];
                stdev[item][i] = tp.stdev[i];
            }
            item++;
        }

        // print results and average values
        for( int i=0; i<params.length; i++ ) {
            double sum=0;
            System.out.format("%-10s, avg",params[i]);
            String buf="";
            for( int j=0; j<dji30.length; j++ ) {
                sum += avg[j][i];
                buf += String.format(", %.4f", avg[j][i]);
            }
            System.out.format(", %.8f%s\n", sum/dji30.length, buf);
        }
        for( int i=0; i<params.length; i++ ) {
            double sum=0;
            System.out.format("%-10s, stdev",params[i]);
            String buf="";
            for( int j=0; j<dji30.length; j++ ) {
                sum += stdev[j][i];
                buf += String.format(", %.4f", stdev[j][i]);
            }
            System.out.format(", %.8f%s\n", sum/dji30.length, buf);
        }

    }

/*
closeMA200, avg, 165.62033659, 309.9235, 263.5790, 213.1054, 281.2635, 405.6557, 218.2976, 244.8739, 159.9331, 140.8206, 228.7482, 231.2753, 134.0378, 159.4352, 136.3831, 210.7287, 227.7290, 138.1910, 180.7110, 148.4106, 36.5138, 134.4871, 124.7310, 133.6567, 120.4148, 49.2456, 84.7131, 47.6149, 105.2734, 54.6876, 44.1698
closeMA200xo, avg, 4.74292099, 8.1269, 9.4426, 6.9535, 4.8998, 2.2357, 5.6242, 5.9262, 6.9935, 6.6240, 4.1275, 3.0126, 5.5702, 5.1832, 9.4207, 4.2336, -1.8112, 6.0206, 3.0801, 1.6526, 22.8908, -0.3908, 2.3428, 4.2122, 0.7476, 7.4100, 2.1852, 2.9606, -0.0048, 2.8371, -0.2194
closeMA50 , avg, 170.72042470, 329.6951, 279.4947, 223.8331, 289.9902, 410.0491, 227.5434, 254.2232, 168.6049, 148.3489, 235.5279, 236.2571, 140.6540, 165.8673, 143.5802, 216.2663, 221.5511, 143.6687, 184.3932, 150.0059, 42.7402, 133.4083, 126.4983, 137.3152, 120.5602, 52.2268, 85.7820, 48.6357, 104.9300, 55.8825, 44.0793
closeMA50xo, avg, 1.27341681, 2.3939, 2.4036, 2.0606, 1.4536, 0.4889, 1.4128, 1.5579, 1.8445, 1.8071, 0.9500, 0.7731, 1.5928, 1.3759, 2.6305, 1.0415, -0.1600, 1.6665, 0.8182, 0.5211, 5.9732, -0.0097, 0.7328, 0.8785, 0.2520, 1.8377, 0.4289, 0.8004, 0.1653, 0.6205, -0.1095
cmf       , avg, 0.03265432, 0.0363, 0.0741, 0.0406, 0.0527, 0.0170, 0.0432, 0.0290, 0.0474, 0.0446, 0.0338, 0.0289, 0.0537, 0.0360, 0.0660, 0.0401, -0.0219, 0.0375, 0.0205, 0.0384, 0.0736, 0.0199, 0.0020, 0.0326, 0.0147, 0.0238, 0.0129, 0.0468, 0.0030, 0.0241, 0.0085
macd      , avg, 0.52668275, 2.3842, 1.5985, 1.2895, 0.9700, 0.2788, 0.8201, 0.8756, 0.8914, 0.8246, 0.5568, 0.4240, 0.6672, 0.6369, 0.7732, 0.4371, -0.3978, 0.5744, 0.3350, 0.1923, 0.6817, -0.0497, 0.2111, 0.2714, 0.0250, 0.2829, 0.0725, 0.1029, 0.0017, 0.0869, -0.0178
macdSignal, avg, 0.51746509, 2.3323, 1.5902, 1.2538, 0.9636, 0.2402, 0.8156, 0.8798, 0.8754, 0.8095, 0.5585, 0.4197, 0.6444, 0.6270, 0.7523, 0.4387, -0.4087, 0.5728, 0.3323, 0.1795, 0.6758, -0.0560, 0.2063, 0.2741, 0.0234, 0.2799, 0.0715, 0.1011, 0.0018, 0.0878, -0.0188
atrDaily  , avg, 3.88681522, 8.2511, 6.1076, 5.7057, 6.2120, 9.6364, 4.5889, 5.7393, 4.1637, 3.2822, 3.9494, 5.2221, 2.8943, 3.3609, 3.4875, 6.1889, 7.0736, 3.9521, 3.6383, 2.4454, 1.8455, 2.9135, 2.9286, 2.2505, 2.9042, 0.9381, 1.7257, 0.9675, 2.5848, 0.8913, 0.7553
atr       , avg, 3.88247562, 8.2125, 6.0936, 5.6839, 6.2092, 9.6343, 4.5846, 5.7340, 4.1507, 3.2741, 3.9472, 5.2175, 2.8812, 3.3567, 3.4800, 6.1831, 7.0939, 3.9505, 3.6399, 2.4462, 1.8313, 2.9186, 2.9304, 2.2512, 2.9077, 0.9351, 1.7234, 0.9683, 2.5866, 0.8918, 0.7567
atrPct    , avg, 2.30352590, 2.5236, 2.2281, 2.5982, 2.1659, 2.4404, 2.0611, 2.3056, 2.5460, 2.2836, 1.7119, 2.2027, 2.0233, 2.0823, 2.4545, 2.9105, 3.4731, 2.7659, 2.0271, 1.6425, 4.1765, 2.2402, 2.4165, 1.6543, 2.4474, 1.7634, 2.0323, 2.0291, 2.5327, 1.6196, 1.7476
mfi       , avg, 52.80884259, 53.7944, 53.8597, 53.2504, 52.8498, 51.6892, 53.5332, 53.6127, 52.8983, 54.6253, 52.9094, 50.7632, 54.4156, 54.1174, 54.6774, 52.7472, 51.6562, 51.4294, 52.6984, 51.3149, 55.7024, 50.9495, 52.6652, 54.4798, 49.6890, 54.1294, 51.6607, 53.4565, 50.9314, 52.8449, 50.9142
pvo       , avg, -0.69947629, -1.0087, -0.7196, -0.8632, -0.8216, -0.5819, -0.7860, -0.3567, -0.7929, -0.9370, -0.6355, -0.5960, -0.9215, -0.6861, -0.5271, -1.2075, -0.9203, -0.2606, -0.5010, -0.9510, 0.0702, -1.1528, -0.6341, -0.5322, -0.9926, -0.6200, -0.5687, -0.6926, -0.8100, -0.4726, -0.5049
obv       , avg, 3.74872051, 4.4757, 6.2945, 3.5071, 4.8358, 3.8805, 6.4776, 3.5722, 2.0920, 4.3452, 3.1964, 1.2129, 5.6417, 7.1122, 4.7640, 3.8525, 0.9564, 4.5184, 4.8808, 2.7089, 9.3626, -0.2877, 2.9541, 6.3393, -1.3095, 5.3201, 2.9352, 4.3832, 0.1196, 4.4001, -0.0801
willR     , avg, 44.20454974, 42.0390, 39.1478, 43.9364, 43.0369, 44.9953, 40.1426, 42.2998, 41.9740, 41.1903, 42.9109, 47.6156, 43.6328, 42.0781, 39.7286, 45.8686, 50.0119, 43.7016, 45.4225, 46.0660, 39.1521, 47.5694, 45.0963, 43.2405, 50.7264, 42.5354, 46.6572, 43.7832, 47.6792, 44.7008, 49.1976
kcLwr     , avg, 164.12504795, 318.5351, 270.8932, 215.3012, 279.7685, 391.3232, 220.2097, 244.7547, 162.2729, 143.6231, 228.8895, 226.7699, 136.3311, 160.5707, 138.3288, 204.8863, 206.4620, 137.0672, 177.8588, 145.5255, 40.6077, 127.4518, 121.1065, 133.4244, 114.7954, 50.9861, 82.4973, 46.9276, 99.7662, 54.2948, 42.5222
kcMid     , avg, 171.88999921, 334.9600, 283.0804, 226.6689, 292.1870, 410.5917, 229.3790, 256.2226, 170.5743, 150.1712, 236.7840, 237.2050, 142.0936, 167.2840, 145.2887, 217.2526, 220.6499, 144.9682, 185.1386, 150.4178, 44.2703, 133.2890, 126.9673, 137.9267, 120.6107, 52.8564, 85.9442, 48.8643, 104.9394, 56.0784, 44.0357
KcUpr     , avg, 179.65495046, 351.3849, 295.2676, 238.0367, 304.6055, 429.8602, 238.5482, 267.6905, 178.8757, 156.7194, 244.6784, 247.6400, 147.8560, 173.9973, 152.2487, 229.6189, 234.8378, 152.8693, 192.4184, 155.3102, 47.9328, 139.1262, 132.8281, 142.4291, 126.4261, 54.7266, 89.3911, 50.8010, 110.1126, 57.8619, 45.5491
kcLPct    , avg, -4.90150646, -5.7507, -5.2376, -5.7521, -4.7230, -4.8550, -4.5604, -5.0267, -5.5749, -5.0889, -3.6900, -4.5740, -4.5209, -4.5623, -5.7332, -5.9337, -6.2393, -5.9224, -4.2366, -3.4363, -9.9921, -4.3134, -4.8834, -3.5731, -4.7324, -4.1434, -4.1285, -4.2321, -4.8541, -3.3929, -3.3818
kcMPct    , avg, 0.30185095, 0.4852, 0.5937, 0.3675, 0.3773, 0.1952, 0.4306, 0.3469, 0.4607, 0.5154, 0.3211, 0.1309, 0.4083, 0.3491, 0.5995, 0.2309, 0.0131, 0.3407, 0.2584, 0.1638, 0.6589, 0.0429, 0.1650, 0.2932, 0.0408, 0.4805, 0.1410, 0.2757, 0.1173, 0.2518, 0.0001
kcUPct    , avg, 4.29372565, 4.3032, 3.6348, 4.5956, 3.9172, 4.9019, 3.6586, 4.1647, 4.5906, 4.0319, 3.1422, 4.2237, 3.5501, 3.7405, 4.0464, 5.7035, 7.7370, 5.1168, 3.8626, 3.1212, 6.5590, 4.6472, 4.7612, 3.0309, 5.0609, 2.8816, 3.9965, 3.8668, 5.2749, 3.0851, 3.6053
macdv     , avg, 19.92565510, 33.0398, 40.4821, 24.7905, 25.1929, 11.6145, 28.1999, 25.0850, 30.9407, 34.4151, 21.3630, 9.0581, 26.2862, 23.0827, 39.0135, 15.1270, -0.4937, 22.6421, 16.2313, 10.2371, 45.1034, 0.6899, 9.7755, 20.6244, 3.3784, 33.5648, 9.3386, 17.2742, 6.6025, 16.5361, -1.4261
macdvSignal, avg, 19.73303331, 32.4329, 40.3183, 24.3881, 25.0576, 11.1588, 28.0909, 25.1126, 30.5668, 33.9848, 21.3981, 8.9787, 25.7927, 22.7736, 38.3018, 15.1525, -0.5093, 22.5799, 16.1591, 9.8177, 44.6699, 0.5412, 9.6163, 20.7740, 3.3472, 33.2477, 9.2995, 17.1295, 6.6452, 16.6589, -1.4942
mPhase    , avg, 49.90514991, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051, 49.9051
mDir      , avg, 0.52910053, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291, 0.5291
closeMA200, stdev, 238.99458530, 453.8587, 386.9044, 311.8593, 403.0542, 584.2646, 312.5484, 352.5581, 232.8644, 203.8834, 326.5096, 329.4664, 193.5707, 229.2924, 201.1277, 301.9479, 328.8470, 199.0410, 256.8762, 210.4019, 66.0492, 191.9989, 178.6756, 190.5169, 172.1919, 71.5002, 121.1700, 67.5478, 150.8496, 77.6996, 62.7616
closeMA200xo, stdev, 14.26543601, 19.0013, 17.2108, 16.4552, 11.8674, 13.2210, 10.7816, 13.9155, 16.7235, 16.7834, 8.6558, 9.3695, 12.2510, 12.4739, 19.2737, 16.8617, 19.5617, 17.7502, 10.7436, 5.7067, 44.9345, 12.4198, 13.0210, 9.2774, 15.2928, 13.3471, 10.4360, 11.4899, 14.0989, 7.5776, 7.4602
closeMA50 , stdev, 246.80373607, 486.3468, 409.7890, 328.3740, 415.4812, 590.6698, 326.2652, 365.8642, 246.4846, 216.1828, 336.0578, 336.7019, 204.5089, 239.1986, 210.9864, 310.3642, 320.6509, 207.3521, 262.1905, 212.6586, 76.9605, 190.5687, 181.3818, 195.4319, 172.8077, 76.1785, 122.6647, 69.1357, 150.7616, 79.4308, 62.6628
closeMA50xo, stdev, 6.92262897, 8.1934, 6.5474, 8.0900, 6.5519, 7.3577, 5.1649, 6.8289, 7.8516, 7.1035, 4.7867, 5.8585, 6.5252, 5.8990, 8.2247, 8.5428, 11.7957, 8.0673, 5.8561, 4.2083, 15.0778, 6.4551, 7.2616, 4.1388, 8.3423, 5.2337, 5.0508, 5.9844, 8.0490, 4.5120, 4.1197
cmf       , stdev, 0.20682017, 0.2090, 0.2207, 0.2025, 0.2084, 0.2051, 0.2098, 0.2100, 0.2077, 0.2174, 0.1974, 0.2008, 0.2244, 0.2027, 0.2215, 0.2094, 0.2166, 0.1920, 0.2015, 0.2138, 0.2104, 0.2078, 0.1983, 0.2062, 0.1898, 0.1997, 0.2038, 0.1978, 0.2070, 0.2026, 0.2104
macd      , stdev, 3.23695993, 7.8856, 5.0225, 5.1952, 5.0107, 8.1831, 3.0348, 4.6040, 3.5365, 2.8813, 2.8988, 3.7764, 2.8423, 2.4447, 2.9125, 5.0650, 7.1442, 3.0775, 2.7543, 1.6703, 2.2281, 2.2693, 2.2989, 1.4643, 2.6735, 0.8542, 1.2085, 0.7585, 2.2779, 0.6730, 0.4627
macdSignal, stdev, 3.04182862, 7.4620, 4.7743, 4.8723, 4.7324, 7.5944, 2.8217, 4.3534, 3.3265, 2.7211, 2.7174, 3.5374, 2.6352, 2.3061, 2.7296, 4.7749, 6.7233, 2.8953, 2.5758, 1.5622, 2.1326, 2.1227, 2.1374, 1.3662, 2.5344, 0.8023, 1.1301, 0.7161, 2.1404, 0.6298, 0.4275
atrDaily  , stdev, 6.09221636, 13.0864, 9.4998, 8.8423, 9.5037, 15.3716, 7.1198, 8.8532, 6.5979, 5.1649, 6.0811, 8.0813, 4.7368, 5.2096, 5.5117, 9.6704, 11.0421, 6.1391, 5.6088, 3.7207, 3.5648, 4.5368, 4.4970, 3.4564, 4.5875, 1.5211, 2.6597, 1.5012, 4.0598, 1.3763, 1.1645
atr       , stdev, 5.75621837, 12.3718, 9.0083, 8.3808, 9.1122, 14.2080, 6.7685, 8.4062, 6.2222, 4.8856, 5.8145, 7.6108, 4.3693, 4.9360, 5.2243, 9.0434, 10.4998, 5.8322, 5.3345, 3.5606, 3.3308, 4.2577, 4.3073, 3.3143, 4.3090, 1.4261, 2.5069, 1.4126, 3.8061, 1.3172, 1.1097
atrPct    , stdev, 3.42048145, 3.7269, 3.2877, 3.7957, 3.2399, 3.6451, 3.0798, 3.4347, 3.8660, 3.4392, 2.6026, 3.2073, 2.9895, 3.1292, 3.6142, 4.2600, 5.3154, 4.0560, 3.0572, 2.4151, 6.0858, 3.2757, 3.6544, 2.4684, 3.6328, 2.6139, 2.9626, 3.0060, 3.7579, 2.4346, 2.5610
mfi       , stdev, 76.37326945, 77.8797, 77.6630, 77.3620, 76.6001, 74.6417, 77.1978, 77.4796, 76.3636, 78.9265, 76.2988, 73.6836, 78.6598, 77.8148, 78.9484, 76.3626, 75.0260, 74.2938, 76.2496, 74.3364, 80.2889, 73.9066, 76.1945, 78.6988, 72.2782, 78.1720, 74.7878, 77.0774, 73.9291, 76.3827, 73.6943
pvo       , stdev, 8.27405592, 8.0202, 6.5721, 7.7746, 7.9372, 9.8394, 7.5411, 7.9843, 7.7035, 7.8113, 7.4304, 7.2767, 9.1121, 7.0228, 7.7019, 10.9681, 10.3465, 8.6813, 7.6769, 9.1774, 8.7789, 10.7293, 7.9165, 6.0966, 9.8653, 9.0442, 7.4349, 7.3539, 10.0314, 7.2073, 7.1854
obv       , stdev, 24.64325854, 26.6645, 25.1119, 26.6176, 26.9170, 22.7685, 23.8082, 23.9255, 23.3784, 25.9073, 23.8811, 24.4927, 26.9583, 25.1096, 24.4848, 24.2996, 26.1217, 22.7415, 23.1160, 24.3964, 25.5978, 24.8603, 24.3199, 24.5382, 24.0548, 24.8088, 24.4621, 24.2767, 26.3444, 22.8633, 22.4708
willR     , stdev, 69.46943854, 66.7898, 63.2170, 69.2228, 68.2524, 70.1006, 63.9872, 66.9600, 66.4888, 65.7395, 67.5918, 73.7821, 69.0516, 66.2755, 64.0214, 71.5750, 77.0439, 68.8130, 71.0115, 71.8884, 62.9908, 73.9658, 70.5969, 68.0532, 77.9722, 67.1894, 72.5398, 68.9532, 74.3944, 70.0272, 75.5880
kcLwr     , stdev, 237.48339324, 471.0693, 397.5178, 316.3793, 400.9423, 564.1007, 315.9778, 352.3889, 237.6697, 209.7801, 326.6992, 323.2265, 198.3390, 231.7823, 203.2580, 294.1978, 299.6821, 197.9830, 253.0175, 206.3637, 73.2692, 182.1863, 173.7962, 189.8913, 164.6640, 74.4200, 118.0002, 66.7618, 143.4669, 77.2000, 60.4709
kcMid     , stdev, 248.59394312, 495.0862, 414.9825, 332.8156, 418.5979, 591.4791, 328.9260, 368.6635, 249.6026, 219.1528, 337.8543, 338.0773, 206.7836, 241.3564, 213.3440, 311.7661, 319.4872, 209.2747, 263.2442, 213.2675, 79.6668, 190.4233, 182.0875, 196.2573, 172.9352, 77.1613, 122.8922, 69.4891, 150.8229, 79.7164, 62.6046
KcUpr     , stdev, 259.74081682, 519.1714, 432.4925, 349.2829, 436.3003, 618.9510, 341.9198, 384.9844, 261.5832, 228.5588, 349.0404, 352.9597, 215.2517, 250.9535, 223.4634, 329.3905, 339.4360, 220.6051, 273.5046, 220.1851, 86.1046, 198.6837, 190.4075, 202.6401, 181.2387, 79.9103, 127.7937, 72.2240, 158.2032, 82.2403, 64.7441
kcLPct    , stdev, 7.77222570, 8.9385, 7.9603, 9.0625, 7.3833, 7.9478, 6.9854, 7.8707, 8.7902, 7.9083, 5.8047, 7.2485, 7.1941, 7.1083, 8.7900, 9.4065, 11.0923, 9.2791, 6.7208, 5.4060, 15.2451, 7.0407, 8.0646, 5.5466, 7.9431, 6.5066, 6.4681, 6.7049, 8.0231, 5.3331, 5.3936
kcMPct    , stdev, 1.48703185, 1.5978, 1.6140, 1.5760, 1.5745, 1.4371, 1.4017, 1.5043, 1.5614, 1.5973, 1.4143, 1.3695, 1.6323, 1.3460, 1.7186, 1.4679, 1.5303, 1.4571, 1.4316, 1.3291, 1.7428, 1.4837, 1.3805, 1.3450, 1.5021, 1.4971, 1.3520, 1.4784, 1.6072, 1.3694, 1.2920
kcUPct    , stdev, 7.92920582, 8.3358, 6.6960, 8.3344, 7.4305, 9.2353, 6.6738, 7.6288, 9.1192, 7.9859, 6.0322, 7.0086, 6.7762, 7.0878, 7.7202, 9.9968, 16.1620, 9.1750, 7.2498, 5.2903, 12.4007, 7.9261, 9.0403, 5.3171, 9.0640, 5.2521, 6.7108, 6.9574, 9.5491, 5.7981, 5.9219
macdv     , stdev, 82.23265115, 89.8310, 93.3445, 88.1981, 87.6592, 77.3276, 75.3545, 84.8007, 87.3364, 90.3101, 78.5927, 74.7894, 91.2356, 74.9228, 99.3188, 81.2247, 83.7464, 78.4206, 77.1093, 70.9492, 103.1828, 81.5094, 72.5446, 72.9342, 86.5344, 84.5651, 71.3335, 82.6258, 88.4279, 74.2166, 64.6337
macdvSignal, stdev, 77.78983534, 85.2837, 89.9185, 83.2848, 83.1448, 72.0347, 71.4938, 80.8700, 83.0559, 86.0087, 74.7401, 70.2050, 85.7506, 71.5188, 95.2152, 76.8933, 78.5209, 73.9853, 72.5233, 66.6203, 99.6570, 76.1632, 67.7241, 68.9456, 82.0626, 80.1173, 66.7394, 78.2786, 82.9848, 69.9464, 60.0084
mPhase    , stdev, 78.88413791, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841, 78.8841
mDir      , stdev, 100.03080758, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308, 100.0308


 */
public static void trainingSet(String tkr, String DB, String NET, String OUT, String[] filters, String[] params, int multiplier, int iterationsNum, int neurons) {
    // get scaling factors
    TrainingProcessor tp = new TrainingProcessor();
    //String[] filters = {"2017", "2018", "2019", "2021", "2022", "2023"};
    //String[] params =  {"cmf", "obv", "willR","kcMPct", "kcUPct", "macdv", "macdvSignal"};
    //        String[] params = {"cmf", "macd", "macdSignal", "obv", "pvo", "mfi", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
    /*for (String tkr : dji30a)*/

        String kpiFile = DB + tkr + "_kpis.txt";
        String outFile1 = OUT + tkr + "_out1.txt";
        String outFile2 = OUT + tkr + "_out2.txt";
        String outFile3 = OUT + tkr + "_out3.txt";
        String outFile4 = OUT + tkr + "_out4.txt";
        String tsFile = "c:/_db/ts/"+tkr+"_ts_72.txt";
        String netFile1 = NET + tkr + "_network1.txt";
        String netFile2 = NET + tkr + "_network2.txt";

        String[] _params = ("closeMA200,closeMA200xo,closeMA50,closeMA50xo,"+
        "cmf,macd,macdSignal,atrDaily,atr,atrPct,mfi,pvo,obv,willR,"+
        "kcLwr,kcMid,KcUpr,kcLPct,kcMPct,kcUPct,"+
        "macdv,macdvSignal,mPhase,mDir").split(        ",");//24 elements

        tp.loadDataSet(kpiFile, filters, params, 3/*multiplier*/);
        System.out.println("Writing training set..."+tsFile);
        tp.writeTrainingSet(tsFile);
    }

    public static void trainTicker(String tkr, String DB, String NET, String OUT, String[] filters, String[] params, int multiplier, int iterationsNum, int neurons) {
        // get scaling factors
        TrainingProcessor tp = new TrainingProcessor();
        //String[] filters = {"2017", "2018", "2019", "2021", "2022", "2023"};
        //String[] params =  {"cmf", "obv", "willR","kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //        String[] params = {"cmf", "macd", "macdSignal", "obv", "pvo", "mfi", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
        /*for (String tkr : dji30a)*/ {
            String kpiFile = DB + tkr + "_kpis.txt";
            String outFile1 = OUT + tkr + "_out1.txt";
            String outFile2 = OUT + tkr + "_out2.txt";
            String outFile3 = OUT + tkr + "_out3.txt";
            String outFile4 = OUT + tkr + "_out4.txt";
            String tsFile = OUT + tkr + "_ts_2021.txt";
            String netFile1 = NET + tkr + "_network1.txt";
            String netFile2 = NET + tkr + "_network2.txt";

            tp.loadDataSet(kpiFile, filters, params, multiplier);
            tp.writeTrainingSet(tsFile);

            //deeplayer network
            int inputSize = tp.inputVector[0].length;
            double learningRate = 0.02;
            //int iterationsNum = ITERATIONS;
            DeepLayer nn1 = new DeepLayer(inputSize, neurons, 1);
            System.out.println("Training BUY signals started for ticker="+tkr);
            TrainingProcessor.train(nn1, tp.buySignal, tp.inputVector, learningRate, iterationsNum, outFile1);
            nn1.writeTopology("buySignal, "+kpiFile,netFile1,params,multiplier,iterationsNum,learningRate);
            DeepLayer nn2 = new DeepLayer(inputSize, neurons, 1);
            System.out.println("Training SELL signals started for ticker="+tkr);
            TrainingProcessor.train(nn2, tp.sellSignal, tp.inputVector, learningRate, iterationsNum, outFile2);
            nn2.writeTopology("sellSignal, "+kpiFile,netFile2,params,multiplier,iterationsNum,learningRate);

            tp.writePredictions( nn1, tp.buySignal, nn2, tp.sellSignal, outFile3 );

            String[] filtersPred = {"2024", "2025"};
            TrainingProcessor tp2 = new TrainingProcessor();
            tp2.loadDataSet(kpiFile, filtersPred, params, multiplier);
            tp2.writePredictions( nn1, tp2.buySignal, nn2, tp2.sellSignal, outFile4 );
            //break; //- debug only
        }
    }


    public static void main_04(String[] args) {
        // get scaling factors
        TrainingProcessor tp = new TrainingProcessor();
        String[] params =  {"cmf", "obv", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
//       String[] params = {"cmf", "macd", "macdSignal", "obv", "pvo", "mfi", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
        for (String tkr : dji30a) {
            String kpiFile = "c:/_db/" + tkr + "_kpis.txt";
            String outFile4 = "c:/_arcturus/neural/" + tkr + "_out4.txt";
            String netFile1 = "c:/_arcturus/neural/" + tkr + "_network1.txt";
            String netFile2 = "c:/_arcturus/neural/" + tkr + "_network2.txt";

            DeepLayer nn1 = new DeepLayer(24, 1024, 1);
            nn1.readTopology(netFile1);
            DeepLayer nn2 = new DeepLayer(24, 1024, 1);
            nn2.readTopology(netFile2);

            String[] filtersPred = {"2024", "2025"};
            TrainingProcessor tp2 = new TrainingProcessor();
            tp2.loadDataSet(kpiFile, filtersPred, params, 3);
            tp2.writePredictions( nn1, tp2.buySignal, nn2, tp2.sellSignal, outFile4 );
            break; //-debug
        }
    }


    public static void main_05(String tkr, String DB, String NET, String OUT, String[] models, String[] filters, String[] params, int multiplier) {
        // get scaling factors
       // TrainingProcessor tp = new TrainingProcessor();
       // String[] params =  {"cmf", "obv", "willR","kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //String[] params =  {"cmf", "macd", "macdSignal", "obv", "pvo", "mfi", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //String[] newParams =  {"cmf", "obv", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
    /* results:
     0, 0.000, 0.000, 0.000, 0.000
 1, *cmf -1.209, -5.085, 10.714, 0.000
 2, 1.754, 11.221, 0.000, 5.263
 3, 1.754, 16.667, 0.000, 0.000
 4, *obv -17.642, 47.368, 17.857, -47.368
 5, 1.754, 30.031, 0.000, -21.053
 6, 1.754, -1.754, 0.000, -31.579
 7, *willR -13.289, -1.754, -35.714, -57.895
 8, *kcLPct-0.412, -1.754, -10.714, -5.263
 9, *kcMPct -45.489, -49.183, -82.143, -73.684
10, kcUPct 1.688, -55.878, -3.571, 31.579
11, *macdv 11.538, -0.170, -50.000, 10.526
12, *macdvSignal0.772, -18.129, -35.714, -47.368
     */
        StringBuilder results = new StringBuilder();

        int columns = params.length;

        /*for (String tkr : dji30a)*/ {
            String kpiFile = DB + tkr + "_kpis.txt";
            String netFile1 = NET + tkr + "_network1.txt";
            String netFile2 = NET + tkr + "_network2.txt";

            DeepLayer nn1 = new DeepLayer(35, 1024, 1);
            nn1.readTopology(netFile1);
            DeepLayer nn2 = new DeepLayer(35, 1024, 1);
            nn2.readTopology(netFile2);

            String[] filtersPred = /*{"2021", "2022", "2023"};//*/{"2024", "2025"};

            double[] bsRecall = new double[columns+1];
            double[] ssRecall = new double[columns+1];
            double[] bsPrecision = new double[columns+1];
            double[] ssPrecision = new double[columns+1];

            for( int knockOut=0; knockOut<=columns; knockOut++ ) {
                TrainingProcessor tp2 = new TrainingProcessor();
                tp2.loadDataSet(kpiFile, filtersPred, params, 5);

                if(knockOut>0) {
                    System.out.println("Knocking out column num=" + knockOut);
                    tp2.knockOutColumn(knockOut-1, columns, 5);
                }
                String tsFile = OUT+String.format("%s_ko_%d_ts_5_2021.txt",tkr,knockOut);
                tp2.writeTrainingSet(tsFile);

                System.out.println("input width="+tp2.inputVector[0].length);

                String outFile5 = OUT+String.format("%s_ko_%d_5_out5.txt",tkr,knockOut);
                tp2.writePredictions(nn1, tp2.buySignal, nn2, tp2.sellSignal, outFile5);
                bsRecall[knockOut] = tp2.buySignalRecall;
                ssRecall[knockOut] = tp2.sellSignalRecall;
                bsPrecision[knockOut] = tp2.buySignalPrecision;
                ssPrecision[knockOut] = tp2.sellSignalPrecision;
            }

            System.out.println("\nFINAL RESULTS for "+tkr+": Input Sensitivity Analysis");
            results.append("\nFINAL RESULTS for "+tkr+": Input Sensitivity Analysis");
            for( int i=0; i<=columns; i++ ) {
                System.out.format("\n%2d, %.3f, %.3f, %.3f, %.3f",i,bsRecall[i],ssRecall[i],bsPrecision[i],ssPrecision[i]);
                results.append( String.format("\n%2d, %.3f, %.3f, %.3f, %.3f",i,bsRecall[i],ssRecall[i],bsPrecision[i],ssPrecision[i]) );
            }
            System.out.println();
            results.append("\n");

            System.out.println("\nFINAL RESULTS for "+tkr+": Input Sensitivity Analysis - Fading Percentages");
            results.append("\nFINAL RESULTS for "+tkr+": Input Sensitivity Analysis - Fading Percentages");
            for( int i=0; i<=columns; i++ ) {
                System.out.format("\n%2d, %.3f, %.3f, %.3f, %.3f",i,
                        (bsRecall[i]/bsRecall[0]-1)*100,
                        (ssRecall[i]/ssRecall[0]-1)*100,
                        (bsPrecision[i]/bsPrecision[0]-1)*100,
                        (ssPrecision[i]/ssPrecision[0]-1)*100);
                results.append( String.format("\n%2d, %.3f, %.3f, %.3f, %.3f",i,
                        (bsRecall[i]/bsRecall[0]-1)*100,
                        (ssRecall[i]/ssRecall[0]-1)*100,
                        (bsPrecision[i]/bsPrecision[0]-1)*100,
                        (ssPrecision[i]/ssPrecision[0]-1)*100) );
            }
            System.out.println();
            results.append("\n");
            
            //break; //-debug only
        }
        String resultsFile = OUT+"sensitivity_analysis_5.txt";
        Utilities.writeFile(resultsFile, results);


    }

    public static void backtestTicker(String tkrSignal, String DB, String NET, String OUT,
                                      String[] models, String[] filters, String[] params, int multiplier, int neurons){
        //String[] params =  {"cmf", "obv", "willR","kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //String[] params =  {"cmf", "obv", "willR","kcLPct", "kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //String[] params =  {"cmf", "obv", "willR","kcMPct", "kcUPct", "macdv", "macdvSignal"};
        //String[] filtersPred = {"2022","2023","2024", "2025"};
       //old code:
        // TrainingProcessor tp = new TrainingProcessor();
        String kpiFile = DB + tkrSignal + "_kpis.txt";
        String outFile = OUT+String.format("%s_backtesting.txt",tkrSignal);
        String resultsFile = OUT+String.format("%s_forecast.txt",tkrSignal);
        //tp.loadDataSet(kpiFile, filters, params, multiplier);

        BacktestingProcessor bp = new BacktestingProcessor( tkrSignal, String.format("simple backtest for %s",tkrSignal) );
        bp.loadDataSet(kpiFile, filters, params, multiplier);
        int totalDays = bp.stockData.samples;
        //int inputSize = bp.stockData.inputVector[0].length;
        //System.out.println("Input size="+inputSize);

        StringBuilder sb = new StringBuilder();

        ModelMixer mx = new ModelMixer(2*models.length, totalDays);
        for (int i = 0; i < models.length; i++) {
            String netFile1 = NET + models[i] + "_network1.txt";
            String netFile2 = NET + models[i] + "_network2.txt";
            mx.loadPredictions( bp, i, String.format("buy[%s]",models[i]), netFile1, neurons);
            mx.loadPredictions(bp,models.length + i, String.format("sell[%s]",models[i]), netFile2, neurons);
        }
        mx.recalculateSignals();
        mx.updateSignalPattern();
        mx.writeForecastDetailMatrix(bp,sb);

        Document doc=bp.backtesting( mx.signals[0], mx.signals[1] );
        sb.append( doc.toJson() );
        System.out.print( doc.toJson() );
        Utilities.writeFile(outFile,sb);
    }
/*
        double[][] buySignal = new double[models.length+1][totalDays];
        double[][] sellSignal = new double[models.length+1][totalDays];
        int j=0;
        for (String tkrModel : models){
            j++;
            String netFile1 = NET + tkrModel + "_network1.txt";
            String netFile2 = NET + tkrModel + "_network2.txt";

            DeepLayer nn1 = new DeepLayer(inputSize, neurons, 1);
            nn1.readTopology(netFile1);
            DeepLayer nn2 = new DeepLayer(inputSize, neurons, 1);
            nn2.readTopology(netFile2);

            for (int d = 0; d < totalDays; d++) {
                double[] y1 = nn1.feedForward(bp.stockData.inputVector[d]);
                double[] y2 = nn2.feedForward(bp.stockData.inputVector[d]);
                buySignal[j][d] = y1[0];
                sellSignal[j][d] = y2[0];
            }
        }*/

 /*       // create detailed output of the summary signals with pricing
        StringBuilder results = new StringBuilder();
        StringBuilder sb3 = new StringBuilder();
        StringBuilder sb4 = new StringBuilder();
        for( String tk : models ) {
            sb3.append( String.format(", b[%s]",tk) );
            sb4.append( String.format(", s[%s]",tk) );
        }
        results.append( String.format("DATES,PRICE,ZIGZAG,buy[Exp],sell[Exp],buy[Act],sell[Act]%s%s",sb3.toString(),sb4.toString()) );
        char[] buyVector = new char[totalDays];
        char[] sellVector = new char[totalDays];
        int maxSig1=0;
        int maxSig2=0;
        for (int d = 0; d < totalDays; d++) {
            int sig1=0;
            int sig2=0;
            StringBuilder sb1 = new StringBuilder();
            StringBuilder sb2 = new StringBuilder();
            for( int i=0; i<models.length; i++ ) {
                if (buySignal[i][d] > 0.8) sig1++;
                if (sellSignal[i][d] > 0.8) sig2++;
                sb1.append(String.format(", %6.3f", buySignal[i][d]));
                sb2.append(String.format(", %6.3f", sellSignal[i][d]));
            }
            buySignal[0][d]=sig1;
            sellSignal[0][d]=sig2;
            results.append( String.format("\n%10s, %.2f, %.2f, %.0f, %.0f, %3d, %3d%s%s",
                    tp.dates[d], tp.price[d], tp.zigZag[d], tp.buySignal[d], tp.sellSignal[d],
                    sig1,sig2,sb1.toString(),sb2.toString()) );
            if( sig1>maxSig1 ) maxSig1=sig1;
            if( sig2>maxSig2 ) maxSig2=sig2;
        }
        String resultsFile = OUT+String.format("%s_forecast.txt",tkrSignal);
        Utilities.writeFile(resultsFile, results);

        // create trading vector
        StringBuilder pattern = new StringBuilder();
        char[] last15days = new char[15];
        maxSig1 = 3*maxSig1/4;
        maxSig2 = 3*maxSig2/4;
        for (int d = 0; d < totalDays; d++) {
            buyVector[d]='0';
            sellVector[d]='0';
            if( buySignal[0][d] >= maxSig1 ) buyVector[d]='1';
            if( sellSignal[0][d] >= maxSig2 ) sellVector[d]='1';

            char flag='-';
            if( buySignal[0][d] > 0 ) flag='b';
            if( sellSignal[0][d] > 0 ) flag='s';
            if( buySignal[0][d] > 0 && sellSignal[0][d] > 0 ) flag='x';

            if( buyVector[d]=='1' && sellVector[d]!='1' ) flag='B';
            if( buyVector[d]!='1' && sellVector[d]=='1' ) flag='S';
            if( buyVector[d]=='1' && sellVector[d]=='1' ) flag='X';
            //if( buyVector[d]!='1' && sellVector[d]!='1' ) pattern.append('-');
            pattern.append( flag );
            if( d>=totalDays-15 ) last15days[d-(totalDays-15)]=flag;
            if( d%90 ==0 ) pattern.append('\n');
        }
*/
       // System.out.format("\nSignal pattern:\n%s\n",pattern);


    public static void optimizeModels(String tkrSignal, String DB, String NET, String OUT, String[] models, String[] filters, String[] params, int multiplier, int neurons) {
        String kpiFile = DB + tkrSignal + "_kpis.txt";
        String outFile = OUT + String.format("%s_optimization.txt", tkrSignal);

        BacktestingProcessor bp = new BacktestingProcessor(tkrSignal, String.format("backtest for %s", tkrSignal));
        bp.loadDataSet(kpiFile, filters, params, multiplier);
        int totalDays = bp.stockData.samples;
        int inputSize = bp.stockData.inputVector[0].length;
        System.out.println("Ticker="+tkrSignal+" Input size=" + inputSize + "  samples=" + totalDays);

        ModelMixer mx = new ModelMixer(2*models.length, totalDays);
        for (int i = 0; i < models.length; i++) {
            String netFile1 = NET + models[i] + "_network1.txt";
            String netFile2 = NET + models[i] + "_network2.txt";
            mx.loadPredictions( bp, i, String.format("buy(%s)",models[i]), netFile1, neurons);
            mx.loadPredictions(bp,models.length + i, String.format("sell(%s)",models[i]), netFile2, neurons);
        }


        StringBuilder sb = new StringBuilder();
        mx.startOptimization(bp);

        //mx.runOptimizer1(bp);
        mx.runOptimizer2(bp, sb);
       //System.out.print(sb);

        sb.append("\n\nFINAL CONFIG:\n");
        mx.writeModelConfig(bp,sb,tkrSignal,neurons);

        mx.writeFinalPredictions(bp,sb);

        sb.append("\n\nCURRENT SIGNAL MATRIX:\n");
        mx.writeSignalsMatrix(sb);
        sb.append("\n\nOPTIMIZED RESULTS:\n");
        Document optimalModel = mx.forecast(bp, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        sb.append( optimalModel.toJson() );
        System.out.println(optimalModel.toJson());

        BacktestingProcessor bp2 = new BacktestingProcessor(tkrSignal, String.format("final backtest for %s", tkrSignal));
        bp2.loadDataSet(kpiFile, new String[]{"2024","2025"}, params, multiplier);

        ModelMixer mx2 = new ModelMixer(2*models.length, bp2.totalDays);
        for (int m = 0; m < models.length; m++) {
            String netFile1 = NET + models[m] + "_network1.txt";
            String netFile2 = NET + models[m] + "_network2.txt";
            mx2.loadPredictions( bp2, m, String.format("by[%s]",models[m]), netFile1, neurons);
            mx2.loadPredictions(bp2,models.length + m, String.format("sl[%s]",models[m]), netFile2, neurons);
        }
        Document finalResult = mx2.forecast(bp2, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        System.out.println( finalResult.toJson() );
        sb.append("\n\nFINAL FORECAST:\n");
        sb.append( finalResult.toJson() );
        sb.append("\nTIME SERIES DATA:\n");
        //mx2.writeFinalPredictions(bp,sb);
        StringBuilder modelConfig = new StringBuilder();
        mx2.writeModelConfig(bp2,modelConfig,tkrSignal,neurons);
        System.out.println("\nMODEL COFNIG:\n"+modelConfig.toString());
        sb.append( modelConfig.toString() );
        mx2.writeForecastDetailMatrix(bp2,sb);
        Utilities.writeFile(outFile, sb);

    }

    private static void optimizeModelsFromFolder(StringBuilder mdb,String tickerId, String DB, String OUT, String modelFolder, String[] filters, String[] params, int multiplier, int neurons) {
        String kpiFile = DB + tickerId + "_kpis.txt";
        String outFile1 = OUT + String.format("%s_%s_opt_signals.txt", tickerId, Utilities.getTimeTag() );
        String outFile2 = OUT + String.format("%s_%s_out2.txt", tickerId, Utilities.getTimeTag() );
        String outFile3 = OUT + String.format("%s_%s_optimization.txt", tickerId, Utilities.getTimeTag() );
        String outFile4 = OUT + String.format("%s_%s_pred_signals.txt", tickerId, Utilities.getTimeTag() );
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        StringBuilder sb3 = new StringBuilder();
        StringBuilder sb4 = new StringBuilder();

        BacktestingProcessor bp = new BacktestingProcessor(tickerId, String.format("backtest for %s", tickerId));
        bp.loadDataSet(kpiFile, filters, params, multiplier);
        /*int totalDays = bp.stockData.samples;
        int inputSize = bp.stockData.inputVector[0].length;
        System.out.println("Ticker="+tickerId+" Input size=" + inputSize + "  samples=" + totalDays);*/

        List<String> modelFiles = Utilities.getFileNames(modelFolder);
        //for(String modelFile : modelFiles) System.out.println(modelFile);

        ModelMixer mx = ModelMixer.createFromModelFiles(modelFiles,neurons,bp);
        /*ModelMixer mx = new ModelMixer(modelFiles.size(), totalDays);
        for (int i = 0; i < modelFiles.size(); i++) {
            System.out.format("Loading optimization model=%s\n", modelFiles.get(i));
            mx.loadPredictions( bp, i, modelFiles.get(i), modelFolder+modelFiles.get(i), neurons);
        }*/

        mx.startOptimization(bp);
        //mx.writeForecastDetailMatrix(bp,sb1);
        //Utilities.writeFile(outFile1, sb1);

        //mx.runOptimizer1(bp);
        mx.runOptimizer2(bp,sb3);
        //System.out.print(sb);

        /*sb2.append("\n\nFINAL CONFIG:\n");
        mx.writeModelConfig(bp,sb2);
        mx.writeFinalPredictions(bp,sb2);
*/
        //sb3.append("\n\nCURRENT SIGNAL MATRIX:\n");
        //mx.writeSignalsMatrix(sb3);
        //mx.writeForecastDetailMatrix(bp,sb3);
        //sb3.append("\n\nOPTIMIZED RESULTS:\n");
        Document optimalModel = mx.forecast(bp, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        sb3.append( optimalModel.toJson() );
        System.out.println(optimalModel.toJson());

        BacktestingProcessor bp2 = new BacktestingProcessor(tickerId, String.format("final backtest for %s", tickerId));
        bp2.loadDataSet(kpiFile, new String[]{"2024","2025"}, params, multiplier);

        ModelMixer mx2 = ModelMixer.createFromModelFiles(modelFiles,neurons,bp2);
        /*ModelMixer mx2 = new ModelMixer(modelFiles.size(), bp2.totalDays);
        for (int i = 0; i < modelFiles.size(); i++) {
            System.out.format("Loading forecast model=%s\n", modelFiles.get(i));
            mx2.loadPredictions( bp, i, modelFiles.get(i), modelFolder+modelFiles.get(i), neurons);
        }*/

        Document finalResult = mx2.forecast(bp2, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        mx2.writeForecastDetailMatrix(bp2,sb4);
        Utilities.writeFile(outFile4, sb4);

        System.out.println( finalResult.toJson() );
        /*sb3.append("\n\nFINAL FORECAST:\n");
        sb3.append( finalResult.toJson() );
        sb3.append("\nTIME SERIES DATA:\n");
        //mx2.writeFinalPredictions(bp,sb);
        mx2.writeForecastDetailMatrix(bp2,sb3);*/
        StringBuilder modelConfig = new StringBuilder();
        mx2.writeModelConfig(bp2,modelConfig,tickerId,neurons);
        System.out.println("\nMODEL COFNIG:\n"+modelConfig.toString());
        sb3.append( modelConfig.toString() );
        Utilities.writeFile(outFile3, sb3);
        mdb.append( modelConfig ).append("\n");
    }

    private static void backtestModels(String tickerId, String KPIS, String OUT, String[] buyModels, String[] sellModels, int buyThreshold, int sellThreshold, String[] periods, String[] config, int multiplier, int neurons) {
        String kpiFile = KPIS + tickerId + "_kpis.txt";
        String outFile = OUT + String.format("%s_%s_models_forecast.txt", tickerId, Utilities.getTimeTag() );
        StringBuilder sb1 = new StringBuilder();

        System.out.println("backtest DLJ models. ticker="+tickerId);

        BacktestingProcessor bp = new BacktestingProcessor(tickerId, String.format("backtest for %s", tickerId));
        bp.loadDataSet(kpiFile, periods, config, multiplier);
        /*int totalDays = bp.stockData.samples;
        int inputSize = bp.stockData.inputVector[0].length;
        System.out.println("Ticker="+tickerId+" Input size=" + inputSize + "  samples=" + totalDays);*/

        List<String> modelFiles = new ArrayList<>();
        char[] modelVector = new char[buyModels.length+sellModels.length];
        int i = 0;
        for( ; i<buyModels.length; i++ ){
            modelFiles.add(buyModels[i]);
            modelVector[i]='B';
        }
        for( int j=0; j<sellModels.length; j++ ){
            modelFiles.add(sellModels[j]);
            modelVector[i++]='S';
        }

        ModelMixer mx = ModelMixer.createFromModelFiles(modelFiles,neurons,bp);

        // Document forecastModel = mx.forecast(bp, modelVector, buyThreshold, sellThreshold);
        // sb1.append( forecastModel.toJson() );
        // System.out.println(forecastModel.toJson());
        mx.recalculateSignals(modelVector, buyThreshold, sellThreshold);
        Document result = bp.backtesting(mx.signals[0], mx.signals[1]);
        //double currentGain = result.getDouble("totalGain");

        sb1.append( result.toJson() );
        System.out.println(result.toJson());

        bp.writeTradingHistoryDetail(sb1,mx);
        mx.writeForecastDetailMatrix(bp,sb1);
        Utilities.writeFile(outFile, sb1);


    }


    private static void backtestTicker2(String tickerId, String DB, String MODEL_FOLDER, String OUT, String[] periods, String[] config, int multiplier, int neurons) {
        String kpiFile = DB + tickerId + "_kpis.txt";
        String outFile = OUT+String.format("%s_backtesting.txt",tickerId);
        String resultsFile = OUT+String.format("%s_forecast.txt",tickerId);
        //tp.loadDataSet(kpiFile, filters, params, multiplier);

        BacktestingProcessor bp = new BacktestingProcessor( tickerId, String.format("simple backtest for %s",tickerId) );
        bp.loadDataSet(kpiFile, periods, config, multiplier);
        int totalDays = bp.stockData.samples;
        int inputSize = bp.stockData.inputVector[0].length;
        System.out.println("Input size="+inputSize);
        StringBuilder sb = new StringBuilder();


        List<String> modelFiles = Utilities.getFileNames(MODEL_FOLDER);
        //for(String modelFile : modelFiles) System.out.println(modelFile);

        ModelMixer mx = new ModelMixer(modelFiles.size()*2, bp.totalDays);
        int numModels = modelFiles.size();
        for (int i = 0; i < numModels; i++) {
            mx.loadPredictions2( bp, i, i+numModels, modelFiles.get(i) );
        }

        mx.recalculateSignals_Majority(numModels);
        mx.updateSignalPattern();
        mx.writeForecastDetailMatrix(bp,sb);
        Document doc=bp.backtesting( mx.signals[0], mx.signals[1] );
        sb.append( doc.toJson() );
        System.out.print( doc.toJson() );
        Utilities.writeFile(outFile,sb);

    }

    private static void optimizeModelsFromFolder2(
            StringBuilder mdb, String tickerId, String DB, String MODEL_FOLDER,
            String OUT, String[] periods, String[] config, int multiplier, int neurons) {

        String kpiFile = DB + tickerId + "_kpis.txt";
       // String outFile1 = OUT + String.format("%s_%s_opt_signals.txt", tickerId, Utilities.getTimeTag() );
       // String outFile2 = OUT + String.format("%s_%s_out2.txt", tickerId, Utilities.getTimeTag() );
        String outFile3 = OUT + String.format("%s_%s_optimization.txt", tickerId, Utilities.getTimeTag() );
        String outFile4 = OUT + String.format("%s_%s_pred_signals.txt", tickerId, Utilities.getTimeTag() );
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        StringBuilder sb3 = new StringBuilder();
        StringBuilder sb4 = new StringBuilder();

        BacktestingProcessor bp = new BacktestingProcessor(tickerId, String.format("backtest for %s", tickerId));
        bp.loadDataSet(kpiFile, periods, config, multiplier);

        List<String> modelFiles = Utilities.getFileNames(MODEL_FOLDER);
        //for(String modelFile : modelFiles) System.out.println(modelFile);
        ModelMixer mx = new ModelMixer(modelFiles.size()*2, bp.totalDays);
        int numModels = modelFiles.size();
        for (int i = 0; i < numModels; i++) {
            System.out.println("Loading model: "+modelFiles.get(i));
            mx.loadPredictions2( bp, i, i+numModels, modelFiles.get(i) );
        }

        mx.startOptimization(bp);
        //mx.writeForecastDetailMatrix(bp,sb1);
        //Utilities.writeFile(outFile1, sb1);
        //mx.runOptimizer1(bp);
        mx.runOptimizer2(bp,sb3);
        //System.out.print(sb);

        /*sb2.append("\n\nFINAL CONFIG:\n");
        mx.writeModelConfig(bp,sb2);
        mx.writeFinalPredictions(bp,sb2);
*/
        //sb3.append("\n\nCURRENT SIGNAL MATRIX:\n");
        //mx.writeSignalsMatrix(sb3);
        //mx.writeForecastDetailMatrix(bp,sb3);
        //sb3.append("\n\nOPTIMIZED RESULTS:\n");
        Document optimalModel = mx.forecast(bp, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        sb3.append( optimalModel.toJson() );
        System.out.println(optimalModel.toJson());

        BacktestingProcessor bp2 = new BacktestingProcessor(tickerId, String.format("final backtest for %s", tickerId));
        bp2.loadDataSet(kpiFile, new String[]{"2024","2025"}, config, multiplier);

        ModelMixer mx2 = new ModelMixer(modelFiles.size()*2, bp2.totalDays);
        for (int i = 0; i < numModels; i++) {
            mx2.loadPredictions2( bp2, i, i+numModels, modelFiles.get(i) );
        }

        Document finalResult = mx2.forecast(bp2, mx.includeVector, mx.buyThreshold, mx.sellThreshold);
        mx2.writeForecastDetailMatrix(bp2,sb4);
        Utilities.writeFile(outFile4, sb4);

        System.out.println( finalResult.toJson() );
        /*sb3.append("\n\nFINAL FORECAST:\n");
        sb3.append( finalResult.toJson() );
        sb3.append("\nTIME SERIES DATA:\n");
        //mx2.writeFinalPredictions(bp,sb);
        mx2.writeForecastDetailMatrix(bp2,sb3);*/
        StringBuilder modelConfig = new StringBuilder();
        mx2.writeModelConfig(bp2,modelConfig,tickerId,neurons);
        System.out.println("\nMODEL CONFIG:\n"+modelConfig.toString());
        sb3.append( modelConfig.toString() );
        Utilities.writeFile(outFile3, sb3);
        mdb.append( modelConfig ).append("\n");

    }
}