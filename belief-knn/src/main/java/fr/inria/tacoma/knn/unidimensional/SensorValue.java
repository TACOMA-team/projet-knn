package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.knn.core.LabelledPoint;

public class SensorValue extends LabelledPoint<Double> {

    public SensorValue(@JsonProperty("sensor") String sensor,
                       @JsonProperty("label") String label,
                       @JsonProperty("timestamp") double timestamp,
                       @JsonProperty("value") Double value) {
        super(sensor, label, timestamp, value);
    }
    public SensorValue(String sensor, String label, double timestamp, Double value,
                       StateSet stateSet) {
        super(sensor, label, timestamp, value, stateSet);
    }

    public SensorValue(LabelledPoint<Double> point) {
        super(point);
    }

    @Override
    public String toString() {
        return "{" +
                "\"value\" : " + getValue() +
                ", \"sensor\" : \"" + sensor + '\"' +
                ", \"label\" : \"" + label + '\"' +
                ", \"timestamp\" : " + timestamp +
                '}';
    }
}
