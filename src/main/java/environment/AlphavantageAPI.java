package environment;

import datamodels.TradeData;
import java.util.ArrayList;
import java.util.List;

public class AlphavantageAPI {

    // PREMIUM KEY:  https://www.alphavantage.co/premium/
//	private static final String ALPHAVANTAGE_API_KEY  = "WKOBZ8KHQFU2125K";



    /* FREE KEYS: "WKOBZ8KHQFU2125K", "UH90NXWLH4FXN0EW"; */
    /* obtained from https://www.alphavantage.co/support/#api-key  by TwinsCom, s.r.o.  martin.lhotak@twinscom.net
     * Docs: https://www.alphavantage.co/documentation/
     * Examples: https://www.alphavantage.co/academy/
     * GIT:
     * - https://github.com/mainstringargs/alpha-vantage-scraper,
     * - https://github.com/mainstringargs?tab=repositories
     * - https://github.com/djbj505/alpha-vantage-java
     * - https://github.com/zackurben/alphavantage (node.js)
     *
     * INDICATORS:
     * https://www.fmlabs.com/reference/default.htm?url=MoneyFlowIndex.htm
     * */

    public AlphavantageAPI() {
    }

    public List<TradeData> retrieveData(String tickerId, boolean allHistory ) {
        List<TradeData> list = new ArrayList<TradeData>();
        HttpClient alphavantage;
        alphavantage = new HttpClient();
        //"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&datatype=csv&symbol=IBM&apikey=WKOBZ8KHQFU2125K"

        String url = String.format("https://%s/query?function=TIME_SERIES_DAILY_ADJUSTED&datatype=csv&symbol=%s&apikey=%s%s",
                Environment.ALPHAVANTAGE_API_URL,
                tickerId,
                Environment.ALPHAVANTAGE_API_KEY,
                allHistory ? "&outputsize=full" : "" );
        System.out.println("URL>"+url);
        //ticker,dt,open,high,low,close,closeadj,volume

        String csv = alphavantage.get(url);
        //System.out.println(csv);
        String[] data = csv.split("\n");
        //System.out.println(data[0]);
        //timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient

        //ticker,dt,open,high,low,close,closeadj,volume
        //        0 1     2    3    4     5       6


        for( int i=1; i<data.length; i++ ) {
            String[] field = data[i].split(",");
            //System.out.println(data[i]);

            if( field.length>=6 ) {
                String date = field[0];
                float open = Float.parseFloat(field[1]);
                float high = Float.parseFloat(field[2]);
                float low = Float.parseFloat(field[3]);
                float close = Float.parseFloat(field[4]);
                float adjClose = Float.parseFloat(field[5]);
                float volume = Float.parseFloat(field[6]);
                float dividend = Float.parseFloat(field[7]);
                float splitCoef = Float.parseFloat(field[8]);

                TradeData record = new TradeData(
                        tickerId,date,
                        open,high,low,close,adjClose,volume,dividend,splitCoef);

                list.add( record );
                //System.out.println( String.format("%05d, %s, %s", i, field[0], field[1] ) );


            }else {
                System.out.println( "SKIPPING:>"+data[i] );
            }

        }
        return list;
    }

}
