package fr.inria.tacoma.knn.experimental.continuous;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.knn.core.LabelledPoint;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Optimization of the Knn for a double value and a Dempster combination.
 */
public class ContinuousDempsterDoubleKnn implements ContinuousKnnBelief<Double> {

    private final double k;
    private final double alpha;
    private final FrameOfDiscernment frame;
    private final List<LabelledPoint<Double>> sortedPoints;
    private final Map<String, Double> gammaProvider;
    private final BiFunction<Double, Double, Double> distance;

    public ContinuousDempsterDoubleKnn(List<LabelledPoint<Double>> points, double k, double alpha,
                             FrameOfDiscernment frame,
                             BiFunction<Double, Double, Double> distance,
                             Map<String,Double> gammaProvider) {
        this(points, k, alpha, frame, distance, gammaProvider, false);
    }

    private ContinuousDempsterDoubleKnn(List<LabelledPoint<Double>> points, double k, double alpha,
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

        int integerK = ((int)this.k) + 1;
        while (endIndex - startIndex < integerK - 1) {
            if(startIndex == 0) {
                endIndex = integerK - 1;
                break;
            }
            else if (endIndex == sortedPoints.size() - 1) {
                startIndex = endIndex - integerK + 1;
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
    public double getK() {
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
    public ContinuousKnnBelief<Double> withAlpha(double newAlpha) {
        return new ContinuousDempsterDoubleKnn(sortedPoints, k, newAlpha, frame, distance,
                gammaProvider, true);
    }

    @Override
    public ContinuousKnnBelief<Double> withK(double newK) {
        return null;
    }

    @Override
    public ContinuousKnnBelief<Double> withAlphaAndK(double newAlpha, double newK) {
        return null;
    }

    @Override
    public MutableMass toMass(Double sensorValue) {
        List<LabelledPoint<Double>> knn = knn(sensorValue);
        LabelledPoint<Double> farthestPoint;
        LabelledPoint<Double> firstPoint = knn.get(0);
        LabelledPoint<Double> lastPoint = knn.get(knn.size() - 1);

        if(sensorValue - firstPoint.getValue() >
                lastPoint.getValue()  - sensorValue ) {
            farthestPoint = firstPoint;
            knn = knn.subList(1, knn.size());
        }
        else {
            farthestPoint = lastPoint;
            knn = knn.subList(0, knn.size() - 1);
        }

        MutableMass massResult = knn.stream()
                .map(p -> getMassFunction(sensorValue, p))
                .reduce(Combinations::dempster).get();
        double remainingK = k - (int)k;
        MassFunction reducedMass = getMassFunction(sensorValue, farthestPoint).discount(1 - remainingK);
        massResult = Combinations.dempster(massResult, reducedMass);
        return massResult;
//        Map<StateSet, Double> optimized = getClustersFullIgnoranceValues(sensorValue, knn, farthestPoint);
//
//
//        MutableMass resultMass = frame.newMass();
//
//        double ignoranceMass = 1;
//        for (StateSet stateSet : optimized.keySet()) {
//            double result = 1;
//            for (Map.Entry<StateSet, Double> entry : optimized.entrySet()) {
//                Double value = entry.getValue();
//                if(!entry.getKey().equals(stateSet)) {
//                    result = result * value;
//                }
//                else {
//                    result = result * (1 - value);
//                }
//
//                ignoranceMass = ignoranceMass * value;
//            }
//            resultMass.set(stateSet, result);
//        }
//        resultMass.set(frame.fullIgnoranceSet(), ignoranceMass);
//
//        resultMass.normalize();
//
//        return resultMass;
    }

    private MutableMass getMassFunction(Double value, LabelledPoint<Double> point) {
        MutableMass mass = frame.newMass();
        double gamma = 1.0 / gammaProvider.get(point.getLabel());
        mass.addToFocal(point.getStateSet(),
                alpha * Math.exp(-distance.apply(value, point.getValue()) * gamma));
        mass.putRemainingOnIgnorance();
        return mass;
    }

    private Map<StateSet, Double> getClustersFullIgnoranceValues(Double sensorValue,
                                                                 List<LabelledPoint<Double>> knn,
                                                                 LabelledPoint<Double> farthestPoint) {
        Map<StateSet, Double> optimized = new HashMap<>();

        for (LabelledPoint<Double> point : knn) {
            double gamma = gammaProvider.get(point.getLabel());
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

        optimized.compute(farthestPoint.getStateSet(), (key, v) -> {
            double gamma = gammaProvider.get(farthestPoint.getLabel());
            double newValue = 1 - (alpha *
                    Math.exp(-distance.apply(sensorValue, farthestPoint.getValue()) / gamma));
            double remainingK = k - (int) k;
                newValue = 1 - (remainingK * (1 - newValue));
//            System.out.print("k=" + remainingK+" ; ");
            if (v == null) {
//                System.out.println("newValue=" + newValue +", v not defined -> " + newValue);
                return newValue;
            } else {
//                System.out.println("newValue=" + newValue +", v=" + v + " -> " + v * newValue);
                return v * newValue;
            }
        });
        return optimized;
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