package fr.inria.tacoma.knn;

import java.util.*;

public class TrainingSet {
    private List<Point> points;
    private Set<String> labels;
    private Map<String, Double> standardDevs;

    public TrainingSet(List<Point> points) {
        this.points = Collections.unmodifiableList(points);
        labels = new HashSet<>();
        points.stream().forEach(p -> labels.add(p.getLabel()));
        labels = Collections.unmodifiableSet(labels);
        standardDevs = new HashMap<>();
        labels.stream().forEach(label -> {
            double[] pointValues = points.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .mapToDouble(Point::getValue).toArray();
            double average = Arrays.stream(pointValues)
                    .average().getAsDouble();
            double squareAverage = Arrays.stream(pointValues).map(x -> x * x)
                    .average().getAsDouble();
            standardDevs.put(label, Math.sqrt(squareAverage - average * average));
        });
        standardDevs = Collections.unmodifiableMap(standardDevs);
    }

    public List<Point> getPoints() {
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
