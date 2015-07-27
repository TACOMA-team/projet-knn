package fr.inria.tacoma.knn.core;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

public class DempsterDoubleKnn implements KnnBelief<Double> {

    private final int k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<LabelledPoint<Double>> sortedPoints;
    private final Map<String, Double> gammaProvider;
    private final BiFunction<Double, Double, Double> distance;

    public DempsterDoubleKnn(List<LabelledPoint<Double>> points, int k, double alpha,
                           FrameOfDiscernment frame,
                           BiFunction<Double, Double, Double> distance,
                           Map<String,Double> gammaProvider) {
        this(points, k, alpha, frame, distance, gammaProvider, false);
    }

    private DempsterDoubleKnn(List<LabelledPoint<Double>> points, int k, double alpha,
                            FrameOfDiscernment frame,
                            BiFunction<Double, Double, Double> distance,
                            Map<String,Double> gammaProvider, boolean sorted) {
        assert alpha > 0;
        this.k = k;
        this.alpha = alpha;
        this.frame = frame;
        this.distance = distance;
        this.gammaProvider = gammaProvider;
        if(sorted) {
            this.sortedPoints = points;
        }
        else {
            this.sortedPoints = new ArrayList<>(points);
            Collections.sort(sortedPoints, (p1, p2) -> Double.compare(p1.getValue(), p2.getValue()));
        }
    }

    private List<LabelledPoint<Double>> knn(Double value) {
        int startIndex = 0, endIndex = sortedPoints.size() - 1 ;
        while(endIndex - startIndex > 1) {
            int index = (startIndex + endIndex) / 2;
            if(sortedPoints.get(index).getValue() >= value) {
                endIndex = index;
            }
            else {
                startIndex = index;
            }
        }

        while (endIndex - startIndex < k) {
            if(startIndex == 0) {
                endIndex = k;
                break;
            }
            else if (endIndex == sortedPoints.size() - 1) {
                startIndex = endIndex - k;
                break;
            }

            Double startValue = sortedPoints.get(startIndex - 1).getValue();
            Double endValue = sortedPoints.get(endIndex + 1).getValue();
            if(value - startValue < endValue - value) {
                startIndex--;
            }
            else {
                endIndex++;
            }
        }

        return sortedPoints.subList(startIndex, endIndex + 1);
    }

    @Override
    public int getK() {
        return k;
    }

    @Override
    public double getAlpha() {
        return alpha;
    }

    @Override
    public Map<String, Double> getGammas() {
        return gammaProvider;
    }

    @Override
    public KnnBelief<Double> withAlpha(double newAlpha) {
        return new DempsterDoubleKnn(sortedPoints, k, newAlpha, frame, distance,
                gammaProvider, true);
    }

    @Override
    public KnnBelief<Double> withK(int newK) {
        return new DempsterDoubleKnn(sortedPoints, newK, alpha, frame, distance,
                gammaProvider, true);
    }

    @Override
    public KnnBelief<Double> withAlphaAndK(double newAlpha, int newK) {
        return new DempsterDoubleKnn(sortedPoints, newK, newAlpha, frame, distance,
                gammaProvider, true);
    }

    @Override
    public MutableMass toMass(Double sensorValue) {
        List<LabelledPoint<Double>> knn = knn(sensorValue);

        ConcurrentHashMap<StateSet, Double> optimized = new ConcurrentHashMap<>();
        for (LabelledPoint<Double> point : knn) {
            double gamma = 1.0 / gammaProvider.get(point.getLabel());
            optimized.compute(point.getStateSet(), (k, v) -> {
                double newValue = 1 - (alpha *
                        Math.exp(-distance.apply(sensorValue, point.getValue()) * gamma));
                if(v == null) {
                    return newValue;
                }
                else {
                    return v * newValue;
                }
            });
        }

        MutableMass resultMass = frame.newMass();

        double ignoranceMass = 1;
        for (StateSet stateSet : optimized.keySet()) {
            double result = 1;
            for (Map.Entry<StateSet, Double> entry : optimized.entrySet()) {
                Double value = entry.getValue();
                if(!entry.getKey().equals(stateSet)) {
                    result = result * value;
                }
                else {
                    result = result * (1 - value);
                }

                ignoranceMass = ignoranceMass * value;
            }
            resultMass.set(stateSet, result);
        }
        resultMass.set(frame.fullIgnoranceSet(), ignoranceMass);

        resultMass.normalize();
        return resultMass;
    }


    @Override
    public MassFunction toMassWithoutValue() {
        MutableMass massFunction = frame.newMass();
        massFunction.putRemainingOnIgnorance();
        return massFunction;
    }


    @Override
    public FrameOfDiscernment getFrame() {
        return frame;
    }


}