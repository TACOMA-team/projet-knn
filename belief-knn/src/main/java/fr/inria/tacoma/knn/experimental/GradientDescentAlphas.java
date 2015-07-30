package fr.inria.tacoma.knn.experimental;

import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.unidimensional.DempsterAlphaDoubleKnn;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class GradientDescentAlphas {

    public static final double DIFF = 0.0001;
    public static final double GRADIENT_STEP = 1;
    public static final double MAX_ALPHA = 0.9;
    public static final double STOP_CRITERION = 0.0000001;


    private SortedMap<String, Double> alphaProvider;
    private final KnnFactory<Double> factory;
    private final List<LabelledPoint<Double>> trainingSet;
    private final List<LabelledPoint<Double>> crossValidation;
    private final SortedSet<String> states;
    private final Map<String, Double> stdDevs;
    private int k;
    public double step = 100;

    public GradientDescentAlphas(KnnFactory<Double> factory,
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
    }

    private double[] computeGrad() {
        double[] grad = new double[alphaProvider.size()];
        //derivative for alphas
        SortedMap<String,Double> alphas2 = new TreeMap<>(alphaProvider);
        int i = 0;
        for (Map.Entry<String, Double> entry : alphaProvider.entrySet()) {
            String label = entry.getKey();
            Double alphaValue = entry.getValue();
            alphas2.put(label, alphaValue + DIFF);
            grad[i] = error(alphas2);
            alphas2.put(label, alphaValue - DIFF);
            grad[i] -= error(alphas2);
            grad[i] /= (2 * DIFF);
            //reset the map
            alphas2.put(label, alphaValue);
            ++i;
        }
        return grad;
    }

    private double error(SortedMap<String, Double> alphas) {
        KnnBelief<Double> model = new DempsterAlphaDoubleKnn(trainingSet, k, alphas,
                factory.getFrame(), factory.getDistance(), stdDevs);
        return KnnUtils.error(crossValidation, model);
    }


    private double iterate() {
        double currentError = error(alphaProvider);
        double newError = Double.MAX_VALUE;
        double[] grad = computeGrad();

        while(newError > currentError) {

            SortedMap<String, Double> newAlphas = getNewAlphas(grad);
            newError = error(newAlphas);

            if (newError < currentError) {
                this.step *= 2;
                alphaProvider = newAlphas;
            } else {
                this.step /= 2;
            }
        }

//        System.out.println("step: " + step);
//        System.out.println("(" + alpha + "," + normalizedGamma + " ) -> " + error(alpha,
//                normalizedGamma));
        return error(alphaProvider);
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
        double currentError = error(alphaProvider);
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
                factory.getFrame(), factory.getDistance(), stdDevs);
    }
}
