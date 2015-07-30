package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Optimization of the Knn for a double value and a Dempster combination.
 */
public class DempsterAlphaDoubleKnn implements KnnBelief<Double> {

    private final int k;
    private final Map<String, Double> alphaProvider;
    private final FrameOfDiscernment frame;
    private final List<LabelledPoint<Double>> sortedPoints;
    private final Map<String, Double> gammaProvider;
    private final BiFunction<Double, Double, Double> distance;

    public DempsterAlphaDoubleKnn(List<LabelledPoint<Double>> points, int k, Map<String, Double> alphaProvider,
                                  FrameOfDiscernment frame,
                                  BiFunction<Double, Double, Double> distance,
                                  Map<String, Double> gammaProvider) {
        this(points, k, alphaProvider, frame, distance, gammaProvider, false);
    }

    private DempsterAlphaDoubleKnn(List<LabelledPoint<Double>> points, int k,
                                   Map<String, Double> alphaProvider,
                                   FrameOfDiscernment frame,
                                   BiFunction<Double, Double, Double> distance,
                                   Map<String, Double> gammaProvider, boolean sorted) {
        this.k = k;
        this.alphaProvider = alphaProvider;
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

    public Map<String, Double> getAlphaProvider() {
        return alphaProvider;
    }

    @Override
    public int getK() {
        return k;
    }

    @Override
    public double getAlpha() {
        return 0;
    }

    @Override
    public Map<String, Double> getGammas() {
        return gammaProvider;
    }

    @Override
    public KnnBelief<Double> withAlpha(double newAlpha) {
        return null;
    }

    @Override
    public KnnBelief<Double> withK(int newK) {
        return null;
    }

    @Override
    public KnnBelief<Double> withAlphaAndK(double newAlpha, int newK) {
        return null;
    }

    @Override
    public MutableMass toMass(Double sensorValue) {
        List<LabelledPoint<Double>> knn = knn(sensorValue);

        Map<StateSet, Double> optimized = new HashMap<>();
        for (LabelledPoint<Double> point : knn) {
            double gamma = gammaProvider.get(point.getLabel());
            double alpha = alphaProvider.get(point.getLabel());
            optimized.compute(point.getStateSet(), (k, v) -> {
                double newValue = 1 - (alpha *
                        Math.exp(-distance.apply(sensorValue, point.getValue()) / gamma));
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
        return (MutableMass)Mass.toConsonant(resultMass);
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