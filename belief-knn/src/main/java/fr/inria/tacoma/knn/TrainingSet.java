package fr.inria.tacoma.knn;

import fr.inria.tacoma.knn.unidimensional.SensorValue;

import java.util.*;

public class TrainingSet {
    private List<SensorValue> points;
    private Set<String> labels;
    private Map<String, Double> standardDevs;

    public TrainingSet(List<SensorValue> points) {
        this.points = Collections.unmodifiableList(points);
        labels = new HashSet<>();
        points.stream().forEach(p -> labels.add(p.getLabel()));
        labels = Collections.unmodifiableSet(labels);
        standardDevs = new HashMap<>();
        labels.stream().forEach(label -> {
            double[] pointValues = points.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .mapToDouble(SensorValue::getValue).toArray();
            double average = Arrays.stream(pointValues)
                    .average().getAsDouble();
            double squareAverage = Arrays.stream(pointValues).map(x -> x * x)
                    .average().getAsDouble();
            standardDevs.put(label, Math.sqrt(squareAverage - average * average));
        });
        standardDevs = Collections.unmodifiableMap(standardDevs);
    }

    public List<SensorValue> getPoints() {
        return points;
    }

    public Set<String> getLabels() {
        return labels;
    }

    public Map<String, Double> getStandardDevs() {
        return standardDevs;
    }

    public int getSize() {
        return points.size();
    }
}
