package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.knn.LabelledPoint;

public class SensorValue extends LabelledPoint {
    private double value;

    public SensorValue(@JsonProperty("sensor") String sensor,
                       @JsonProperty("label") String label,
                       @JsonProperty("timestamp") double timestamp,
                       @JsonProperty("value") double value) {
        super(sensor, label, timestamp);
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "{" +
                "\"value\" : " + value +
                ", \"sensor\" : \"" + sensor + '\"' +
                ", \"label\" : \"" + label + '\"' +
                ", \"timestamp\" : " + timestamp +
                '}';
    }
}
