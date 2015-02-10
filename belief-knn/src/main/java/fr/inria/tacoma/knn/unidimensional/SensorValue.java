package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.knn.LabelledPoint;

public class SensorValue extends LabelledPoint<Double> {

    public SensorValue(@JsonProperty("sensor") String sensor,
                       @JsonProperty("label") String label,
                       @JsonProperty("timestamp") double timestamp,
                       @JsonProperty("value") Double value) {
        super(sensor, label, timestamp, value);
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
