package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Features {
    static Function<List<LabelledPoint<Double>>, List<LabelledPoint<Double>>>
            slidingAverage(int windowSize) {
        return list -> {
            List<LabelledPoint<Double>> transformed = new ArrayList<>();
            for (int i = windowSize; i < list.size(); i++) {
                LabelledPoint<Double> point = list.get(i);
                double newValue = list.subList(i - windowSize, i).stream()
                        .mapToDouble(LabelledPoint::getValue)
                        .average().getAsDouble();
                SensorValue newPoint = new SensorValue(point.getSensor(), point.getLabel(),
                        point.getTimestamp(), newValue);
                newPoint.setStateSet(point.getStateSet());
                transformed.add(newPoint);
            }
            return transformed;
        };
    }

    private static Function<List<LabelledPoint<Double>>, List<LabelledPoint<Double>>>
    slidingMax(int windowSize) {
        return list -> {
            List<LabelledPoint<Double>> transformed = new ArrayList<>();
            for (int i = windowSize; i < list.size(); i++) {
                LabelledPoint<Double> point = list.get(i);
                double newValue = list.subList(i - windowSize, i).stream()
                        .mapToDouble(LabelledPoint::getValue)
                        .max().getAsDouble();
                SensorValue newPoint = new SensorValue(point.getSensor(), point.getLabel(),
                        point.getTimestamp(), newValue);
                newPoint.setStateSet(point.getStateSet());
                transformed.add(newPoint);
            }
            return transformed;
        };
    }

    private static Function<List<LabelledPoint<Double>>, List<LabelledPoint<Double>>>
    slidingMin(int windowSize) {
        return list -> {
            List<LabelledPoint<Double>> transformed = new ArrayList<>();
            for (int i = windowSize; i < list.size(); i++) {
                LabelledPoint<Double> point = list.get(i);
                double newValue = list.subList(i - windowSize, i).stream()
                        .mapToDouble(LabelledPoint::getValue)
                        .min().getAsDouble();
                SensorValue newPoint = new SensorValue(point.getSensor(), point.getLabel(),
                        point.getTimestamp(), newValue);
                newPoint.setStateSet(point.getStateSet());
                transformed.add(newPoint);
            }
            return transformed;
        };
    }

    public static List<LabelledPoint<Double>> log(List<LabelledPoint<Double>> points) {
        return points.stream().map(point -> {
            SensorValue newPoint = new SensorValue(point.getSensor(), point.getLabel(),
                    point.getTimestamp(), Math.log(point.getValue()));
            newPoint.setStateSet(point.getStateSet());
            return newPoint;
        }).collect(Collectors.toList());
    }

    public static List<LabelledPoint<Double>> normalized(List<LabelledPoint<Double>> points) {
        DoubleSummaryStatistics stats = points.stream().mapToDouble(
                LabelledPoint::getValue).summaryStatistics();
        double min = stats.getMin();
        double max = stats.getMax();

        return points.stream().map(p -> {
            LabelledPoint<Double> newPoint = new SensorValue(p);
            newPoint.setValue((p.getValue() - min) / (max - min));
            return newPoint;
        }).collect(Collectors.toList());
    }

    public static List<LabelledPoint<Double>> distToAverage(List<LabelledPoint<Double>> points) {
        double average = points.stream().mapToDouble(LabelledPoint::getValue).average().getAsDouble();

        return points.stream().map(p -> {
            LabelledPoint<Double> newPoint = new SensorValue(p);
            newPoint.setValue(Math.abs(average - p.getValue()));
            return newPoint;
        }).collect(Collectors.toList());
    }
}
