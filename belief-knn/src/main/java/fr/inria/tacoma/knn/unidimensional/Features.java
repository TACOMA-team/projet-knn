package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

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
}
