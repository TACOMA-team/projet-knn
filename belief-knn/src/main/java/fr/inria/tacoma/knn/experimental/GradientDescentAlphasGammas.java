package fr.inria.tacoma.knn.experimental;

import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.unidimensional.DempsterAlphaDoubleKnn;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GradientDescentAlphasGammas {

    public static final double DIFF = 0.0001;
    public static final double MAX_ALPHA = 0.9;
    public static final double STOP_CRITERION = 0.0000001;


    private SortedMap<String, Double> alphaProvider;
    private final KnnFactory<Double> factory;
    private final List<LabelledPoint<Double>> trainingSet;
    private final List<LabelledPoint<Double>> crossValidation;
    private final SortedSet<String> states;
    private final Map<String, Double> stdDevs;
    private SortedMap<String, Double> normalizedGamma;
    private int k;
    public double step = 100;

    public GradientDescentAlphasGammas(KnnFactory<Double> factory,
                                       List<LabelledPoint<Double>> trainingSet,
                                       List<LabelledPoint<Double>> crossValidation,
                                       Map<String, Double> stdDevs, int k) {
        this.factory = factory;
        this.trainingSet = trainingSet;
        this.crossValidation = crossValidation;
        this.alphaProvider = new TreeMap<>();
        this.states = new TreeSet<>(factory.getFrame().getStates());
        for (String state : states) {
            alphaProvider.put(state, 2 * DIFF);
        }
        this.k = k;
        this.stdDevs = stdDevs;
        this.normalizedGamma = new TreeMap<>();
        for (String state : states) {
            normalizedGamma.put(state, 1.0);
        }
    }

    private double[] computeGrad() {
        double[] grad = new double[alphaProvider.size() + normalizedGamma.size()];
        //derivative for alphas
        SortedMap<String,Double> alphas2 = new TreeMap<>(alphaProvider);
        int i = 0;
        for (Map.Entry<String, Double> entry : alphaProvider.entrySet()) {
            String label = entry.getKey();
            Double alphaValue = entry.getValue();
            alphas2.put(label, alphaValue + DIFF);
            grad[i] = error(alphas2, normalizedGamma);
            alphas2.put(label, alphaValue - DIFF);
            grad[i] -= error(alphas2, normalizedGamma);
            grad[i] /= (2 * DIFF);
            //reset the map
            alphas2.put(label, alphaValue);
            ++i;
        }

        //gammas
        SortedMap<String,Double> gammas2 = new TreeMap<>(normalizedGamma);
        for (Map.Entry<String, Double> entry : normalizedGamma.entrySet()) {
            String label = entry.getKey();
            Double gammaValue = entry.getValue();
            gammas2.put(label, gammaValue + DIFF);
            grad[i] = error(alphaProvider, gammas2);
            gammas2.put(label, gammaValue - DIFF);
            grad[i] -= error(alphaProvider, gammas2);
            grad[i] /= (2 * DIFF);
            //reset the map
            gammas2.put(label, gammaValue);
            ++i;
        }

        return grad;
    }

    private double error(SortedMap<String, Double> alphas,
                         SortedMap<String, Double> normalizedGamma) {
        Map<String, Double> actualGammas = new HashMap<>();
        normalizedGamma.forEach((key, value) -> actualGammas.put(key, value * stdDevs.get(key)));
        KnnBelief<Double> model = new DempsterAlphaDoubleKnn(trainingSet, k, alphas,
                factory.getFrame(), factory.getDistance(), actualGammas);
        return KnnUtils.error(crossValidation, model);
    }


    private double iterate() {
        double currentError = error(alphaProvider, normalizedGamma);
        double newError = Double.MAX_VALUE;
        double[] grad = computeGrad();

        while(newError > currentError) {

            SortedMap<String, Double> newAlphas = getNewAlphas(grad);
            SortedMap<String, Double> newGammas = getNewGammas(grad);
            newError = error(newAlphas, newGammas);

            if (newError < currentError) {
                this.step *= 2;
                alphaProvider = newAlphas;
                normalizedGamma = newGammas;
            } else {
                this.step /= 2;
            }
        }

//        System.out.println("step: " + step);
//        System.out.println("(" + alpha + "," + normalizedGamma + " ) -> " + error(alpha,
//                normalizedGamma));
        return error(alphaProvider, normalizedGamma);
    }



    private SortedMap<String, Double> getNewGammas(double[] grad) {
        SortedMap<String, Double> newGammas = new TreeMap<>(this.normalizedGamma);
        AtomicInteger i = new AtomicInteger(alphaProvider.size());
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

    private SortedMap<String, Double> getNewAlphas(double[] grad) {
        SortedMap<String, Double> newAlphas = new TreeMap<>(this.alphaProvider);
        AtomicInteger i = new AtomicInteger(0);
        for (String state : states) {
            Double value = this.alphaProvider.get(state) - (grad[i.get()] * step);
            if (value <= DIFF) {
                value = 2 * DIFF;
            }
            if(value > MAX_ALPHA) {
                value = MAX_ALPHA;
            }
            newAlphas.put(state, value);
            i.incrementAndGet();
        }
        return newAlphas;
    }

    public DempsterAlphaDoubleKnn iterate(final int times) {
        double currentError = error(alphaProvider, normalizedGamma);
        double newError = currentError;
        System.out.println(newError);
        for (int i = 0; i < times; i++) {
            currentError = newError;
            newError = iterate();
            System.out.println(newError);
            if(currentError - newError < STOP_CRITERION) {
                System.out.println("reached stop criterion");
                break;
            }
        }
//        System.out.println();
        return new DempsterAlphaDoubleKnn(trainingSet, k, alphaProvider,
                factory.getFrame(), factory.getDistance(), getGamma());
    }

    public Map<String, Double> getGamma() {
        Map<String, Double> actualGammas = new HashMap<>();
        normalizedGamma.forEach((key, value) -> actualGammas.put(key, value * stdDevs.get(key)));
        return actualGammas;
    }
}
