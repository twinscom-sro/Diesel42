package datamodels;

import java.util.HashMap;

public class TradeData {

    String tickerId;
    public String date;
    public float open;
    public float high;
    public float low;
    public float close;
    public float adjClose;
    public float volume;
    public float dividend;
    public float splitCoef;
    HashMap<String,Double> map;

    public TradeData(String _tickerId, String _date ){
        tickerId= _tickerId;
        date = _date;
        map = new HashMap<String,Double>();
    };

    public TradeData(String _tickerId, String _date,
                     float _open, float _high, float _low, float _close, float _adjClose,
                     float _volume, float _dividend, float _splitCoef) {
        tickerId= _tickerId;
        date = _date;
        open = _open;
        high = _high;
        low = _low;
        close = _close;
        adjClose = _adjClose;
        volume = _volume;
        dividend = _dividend;
        splitCoef = _splitCoef;
        map = new HashMap<String,Double>();
    }

    void setValue( String key, double value ){
        map.put(key, value);
    }

    public String toString() {
        return String.format("%s, %f", date, adjClose);
    }

    public String toString(String key) {
        return String.format("%s, %f", date, map.get(key));
    }

    public String toString(String[] keys) {
        String output = String.format("%s, %f, %f, %f, %f, %f, %f, %f, %f",
                date, open, high, low, close, adjClose, volume, dividend, splitCoef );
        for( String k : keys ) {
            output += String.format(", %f", map.get(k));
        }
        return output;
    }

    public void mergeMap(HashMap<String, Double> map2) {
        for( String i : map2.keySet() ) {
            map.put(i, map2.get(i) );
        }

    }

    public double get(String key) {
        Double y=0.;
        if( map.containsKey(key) ){ y= map.get(key); };
        return y;
    }

}
