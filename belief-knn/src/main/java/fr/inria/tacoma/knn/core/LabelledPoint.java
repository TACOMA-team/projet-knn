package fr.inria.tacoma.knn.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.bft.core.frame.StateSet;

public class LabelledPoint<T> {
    protected String sensor;
    protected String label;
    protected double timestamp;
    protected T value;
    protected StateSet stateSet;

    public LabelledPoint(@JsonProperty("sensor") String sensor, @JsonProperty("label") String label,
                         @JsonProperty("timestamp") double timestamp, T value) {
        this(sensor, label, timestamp, value, null);
    }

    public LabelledPoint(LabelledPoint<T> point) {
        this.sensor = point.sensor;
        this.label = point.label;
        this.timestamp = point.timestamp;
        this.value = point.value;
        this.stateSet = point.stateSet;
    }

    public LabelledPoint(String sensor, String label, double timestamp, T value,
                         StateSet stateSet) {
        this.sensor = sensor;
        this.label = label;
        this.timestamp = timestamp;
        this.value = value;
        this.stateSet = stateSet;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getSensor() {
        return sensor;
    }

    public void setSensor(String sensor) {
        this.sensor = sensor;
    }

    public double getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(double timestamp) {
        this.timestamp = timestamp;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    public StateSet getStateSet() {
        return stateSet;
    }

    public void setStateSet(StateSet stateSet) {
        this.stateSet = stateSet;
    }
}
