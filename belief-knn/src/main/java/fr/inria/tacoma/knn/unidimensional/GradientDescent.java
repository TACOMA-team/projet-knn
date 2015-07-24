package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GradientDescent<T> {

    public static final double DIFF = 0.0001;
    public static final double GRADIENT_STEP = 1;
    public static final double MAX_ALPHA = 0.9;
    public static final double STOP_CRITERION = 0.000001;


    private static double alpha = 0.5;
    private final KnnFactory<T> factory;
    private final List<LabelledPoint<T>> trainingSet;
    private final List<LabelledPoint<T>> crossValidation;
    private final TreeSet<String> states;
    private final Map<String, Double> stdDevs;
    private SortedMap<String, Double> normalizedGamma;
    private int k;
    public double step = 100;

    public GradientDescent(KnnFactory<T> factory,
                                List<LabelledPoint<T>> trainingSet,
                                List<LabelledPoint<T>> crossValidation,
                           Map<String, Double> stdDevs) {
        this.factory = factory;
        this.trainingSet = trainingSet;
        this.crossValidation = crossValidation;
        this.normalizedGamma = new TreeMap<>();
        this.states = new TreeSet<>(factory.getFrame().getStates());
        for (String state : states) {
            normalizedGamma.put(state, 1.0);
        }
        k = trainingSet.size() - 1 ;
        this.stdDevs = stdDevs;
    }

    private <T> double[] computeGrad() {
        double[] grad = new double[normalizedGamma.size() + 1];
        //derivative for alpha
        grad[0] = (error(alpha + DIFF, normalizedGamma) - error(alpha - DIFF, normalizedGamma)) / (2 * DIFF);
        //derivative for gammas
        SortedMap<String,Double> gammas2 = new TreeMap<>(normalizedGamma);
        int i = 1;
        for (Map.Entry<String, Double> entry : normalizedGamma.entrySet()) {
            String label = entry.getKey();
            Double gammaValue = entry.getValue();
            gammas2.put(label, gammaValue + DIFF);
            grad[i] = error(alpha, gammas2);
            gammas2.put(label, gammaValue - DIFF);
            grad[i] -= error(alpha, gammas2);
            grad[i] /= (2 * DIFF);
            //reset the map
            gammas2.put(label, gammaValue);
            ++i;
        }
        return grad;
    }

    private double error(double alpha, SortedMap<String, Double> normalizedGamma) {
        Map<String, Double> actualGammas = new HashMap<>();
        normalizedGamma.forEach((key, value) -> actualGammas.put(key, value * stdDevs.get(key)));
        KnnBelief<T> model = factory.newKnnBelief(trainingSet, actualGammas, k, alpha);
        return KnnUtils.error(crossValidation, model);
    }


    private double iterate() {
        double currentError = error(alpha, normalizedGamma);
        double newError = 10;
        double[] grad = computeGrad();


        while(newError > currentError) {
            double newAlpha = iterateAlpha(grad);

            SortedMap<String, Double> newGammas = getNewGammas(grad);
            newError = error(newAlpha, newGammas);

            if (newError < currentError) {
                this.step *= 2;
                alpha = newAlpha;
                normalizedGamma = newGammas;
            } else {
                this.step /= 2;
            }
        }

//        System.out.println("step: " + step);
//        System.out.println("(" + alpha + "," + normalizedGamma + " ) -> " + error(alpha,
//                normalizedGamma));
        return error(alpha, normalizedGamma);
    }

    private SortedMap<String, Double> getNewGammas(double[] grad) {
        SortedMap<String, Double> newGammas = new TreeMap<>(this.normalizedGamma);
        AtomicInteger i = new AtomicInteger(1);
        for (String state : states) {
            Double value = this.normalizedGamma.get(state)- (grad[i.get()] * step);
            if (value <= DIFF) {
                value = 2 * DIFF;
            }
            newGammas.put(state, value);
            i.incrementAndGet();
        }
        return newGammas;
    }

    private double iterateAlpha(double[] grad) {
        double newAlpha = alpha;
        newAlpha -= step * grad[0];
        if(newAlpha > MAX_ALPHA ) {
            newAlpha = MAX_ALPHA ;
        }
        else if (newAlpha <= DIFF) {
            newAlpha = 2 * DIFF;
        }
        return newAlpha;
    }

    public KnnBelief<T> iterate(final int times) {
        double currentError = error(alpha, normalizedGamma);
        double newError = currentError;
        for (int i = 0; i < times; i++) {
            currentError = newError;
            newError = iterate();
            if(currentError - newError < STOP_CRITERION) {
                System.out.println("reached stop criterion");
                break;
            }
            System.out.println(newError);
        }
        System.out.println();
        return factory.newKnnBelief(trainingSet, getGamma(), k, alpha);
    }

    public static double getAlpha() {
        return alpha;
    }

    public Map<String, Double> getGamma() {
        Map<String, Double> actualGammas = new HashMap<>();
        normalizedGamma.forEach((key, value) -> actualGammas.put(key, value * stdDevs.get(key)));
        return actualGammas;
    }
}
