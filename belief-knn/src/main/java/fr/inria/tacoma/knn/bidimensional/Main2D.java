package fr.inria.tacoma.knn.bidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.unidimensional.SensorValue;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main2D {

    public final String PRESENCE;
    public final String ABSENCE;
    public final String PRESENCE_TEST;
    public final String ABSENCE_TEST;
    public final double ALPHA;
    public final int K;
    public final int FUNC;
    public final double ALPHA_STEP;
    public final int K_STEP;
    public final int K_MAX;
    public final double GAMMA1;
    public final double GAMMA2;
    public final int MODE; // 1 : Montre une courbe pour ALPHA,K ; 2 : erreur en fonc de alpha,k
    public final int OPTION; // 0 : centred val ; 1 : derivation ; 2 : moyenne ; 3 : max ; 4 : min
    public final int OPTION2;// 0 : centred val ; 1 : derivation ; 2 : moyenne ; 3 : max ; 4 : min
    public final int SENSOR_VALUE_CENTER;
    public final int MAX_X;
    public final int MAX_Y;


    public Main2D(String PRESENCE, String ABSENCE, String PRESENCE_TEST, String ABSENCE_TEST,
                      double ALPHA, int K, int FUNC, double ALPHA_STEP, int K_STEP, int K_MAX,
                      double GAMMA1, double GAMMA2,
                      int MODE, int OPTION, int OPTION2,
                      int SENSOR_VALUE_CENTER, int MAX_X, int MAX_Y) {
        this.PRESENCE = PRESENCE_TEST;
        this.ABSENCE = ABSENCE;
        this.PRESENCE_TEST = PRESENCE_TEST;
        this.ABSENCE_TEST = ABSENCE_TEST;
        this.ALPHA = ALPHA;
        this.K = K;
        this.FUNC = FUNC;
        this.ALPHA_STEP = ALPHA_STEP;
        this.K_STEP = K_STEP;
        this.K_MAX = K_MAX;
        this.GAMMA1 = GAMMA1;
        this.GAMMA2 = GAMMA2;
        this.MODE = MODE;
        this.OPTION = OPTION;
        this.OPTION2 = OPTION2;
        this.SENSOR_VALUE_CENTER = SENSOR_VALUE_CENTER;
        this.MAX_X = MAX_X;
        this.MAX_Y = MAX_Y;
    }

    public void main2D() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence") ;

        List<LabelledPoint<Coordinate>> absence = CreateList ("absence", ABSENCE, OPTION, OPTION2) ;
        List<LabelledPoint<Coordinate>> presence = CreateList ("presence", PRESENCE, OPTION, OPTION2) ;
        List<LabelledPoint<Coordinate>> absence_test = CreateList ("absence", ABSENCE_TEST, OPTION, OPTION2) ;
        List<LabelledPoint<Coordinate>> presence_test = CreateList ("presence", PRESENCE_TEST, OPTION, OPTION2) ;

        // extracting cross validation data
        List<LabelledPoint<Coordinate>> crossValidation = new ArrayList<>() ;
        crossValidation.addAll(absence_test) ;
        crossValidation.addAll(presence_test) ;

        //creating training set
        List<LabelledPoint<Coordinate>> trainingSet = new ArrayList<>() ;
        trainingSet.addAll(absence) ;
        trainingSet.addAll(presence) ;

        if (MODE == 1)
            showTheOne(frame, trainingSet, crossValidation) ;
        else if (MODE == 2)
            varAlphaK(frame, trainingSet, crossValidation) ;
        else if (MODE == 3)
            bestAlphaK(frame, trainingSet, crossValidation) ;
        else if (MODE == 4)
            grad(frame, trainingSet, crossValidation, GAMMA1, GAMMA2) ;

    }

    private void showTheOne(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation) {
        KnnBelief<Coordinate> beliefModel = new KnnBelief<Coordinate>(trainingSet, K, ALPHA, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
        double error = KnnUtils.error(crossValidation, beliefModel) ;
        System.out.println("k = " + K) ;
        System.out.println("alpha = " + ALPHA) ;
        System.out.println("error = " + error) ;
        show(frame.toStateSet("presence"),beliefModel) ;
    }

    private void varAlphaK(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation) throws IOException {
        int k_mem = 0 ;
        double alpha_mem = 0. ;
        double error_mem = Double.POSITIVE_INFINITY ;
        File file = new File("Result.txt") ;
        file.createNewFile() ;
        FileWriter writer = new FileWriter(file) ;
                writer.write("k,alpha,error\n") ;
        for (int k = 1 ; k < trainingSet.size() & k <= K_MAX ; k += K_STEP) {
            for (double alpha = ALPHA_STEP ; alpha < 1. ; alpha += ALPHA_STEP) {
                KnnBelief<Coordinate> beliefModel = new KnnBelief<Coordinate>(trainingSet, k, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
                double error = KnnUtils.error(crossValidation, beliefModel) ;
                writer.write(k + "," + alpha + "," + error + "\n") ;
                if (error < error_mem) {
                    k_mem = k ;
                    alpha_mem = alpha ;
                    error_mem = error ;
                }
            }
        }
        System.out.println("k = " + k_mem) ;
        System.out.println("alpha = " + alpha_mem) ;
        System.out.println("error = " + error_mem) ;
        writer.flush() ;
        writer.close() ;
    }

    private void bestAlphaK(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation) throws IOException {
        int k_mem = 0 ;
        double alpha_mem = 0. ;
        double error_mem = Double.POSITIVE_INFINITY ;
        for (int k = 1 ; k < trainingSet.size() & k <= K_MAX ; k += K_STEP) {
            for (double alpha = ALPHA_STEP ; alpha < 1. ; alpha += ALPHA_STEP) {
                KnnBelief<Coordinate> beliefModel = new KnnBelief<Coordinate>(trainingSet, k, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
                double error = KnnUtils.error(crossValidation, beliefModel) ;
                if (error < error_mem) {
                    k_mem = k ;
                    alpha_mem = alpha ;
                    error_mem = error ;
                }
            }
        }
        System.out.println("Dimension = 2 ; Option = " + OPTION + ", " + OPTION2 + " ; Fontion = " + FUNC + " -> k = " + k_mem + " ; alpha = " + alpha_mem + " ; error = " + error_mem) ;
    }

    private void show(StateSet stateSet, KnnBelief<Coordinate> model) {
        ChartPanel chartPanel = JfreeChartDisplay2D.getChartPanel(model, 250, MAX_X,
                250, MAX_Y, stateSet);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }

    // Applique la methode du gradient
    private void grad(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation,
                             double gamma1, double gamma2) throws IOException {
        int k_mem = 1 ;
        double alpha_mem = 0. ;
        double error_mem = Double.POSITIVE_INFINITY ;
        boolean stop = false ;
        int count = 0;
        while (! stop) {
//        System.out.println("begin " + count) ;
            int k0 = k_mem ;
            double alpha0 = alpha_mem ;
            double dk = derivate_k(frame, trainingSet, crossValidation, k_mem, alpha_mem) ;
//       System.out.println("dk = " + dk) ;
            double dalpha = derivate_alpha(frame, trainingSet, crossValidation, k_mem, alpha_mem) ;
//       System.out.println("dalpha = " + dalpha) ;
            int k1 = Math.max(k_mem - ((int) (Math.floor(dk * gamma1))), 1);
            int k2 = Math.max(k_mem - ((int) (Math.ceil(dk * gamma1))), 1);
            double alpha = Math.max(Math.min(alpha_mem - dalpha * gamma2, 0.98), 0.01) ;
            KnnBelief<Coordinate> beliefModel1 = new KnnBelief<Coordinate>(trainingSet, k1, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
            double error1 = KnnUtils.error(crossValidation, beliefModel1) ;
//       System.out.println("error1 = " + error1) ;
            KnnBelief<Coordinate> beliefModel2 = new KnnBelief<Coordinate>(trainingSet, k2, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
            double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
//       System.out.println("error2 = " + error2) ;
            if(error1 < error_mem){
                k_mem = Math.max(k1, 1) ;
                alpha_mem = alpha ;
                error_mem = error1 ;
            }
            if(error2 < error_mem){
                k_mem = Math.max(k2, 1) ;
                alpha_mem = alpha ;
                error_mem = error2 ;
            }
//        System.out.println("k_mem = " + k_mem) ;
//    System.out.println("alpha_mem = " + alpha_mem) ;
            count = count + 1 ;
            if (k_mem == k0 && Math.abs(alpha_mem - alpha0) < 0.01)
                stop = true ;
//        System.out.println("end " + count) ;
        }
        System.out.println("k = " + k_mem + " ; alpha = " + alpha_mem + " ; error = " + error_mem) ;
    }

    private double derivate_k(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation,
                             int k, double alpha) throws IOException {
        KnnBelief<Coordinate> beliefModel1 = new KnnBelief<Coordinate>(trainingSet, k, alpha, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
        double error1 = KnnUtils.error(crossValidation, beliefModel1) ;
        KnnBelief<Coordinate> beliefModel2 = new KnnBelief<Coordinate>(trainingSet, k+1, alpha, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
        double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
        return error2 - error1 ;
    }

    private double derivate_alpha(FrameOfDiscernment frame, List<LabelledPoint<Coordinate>> trainingSet, List<LabelledPoint<Coordinate>> crossValidation,
                             int k, double alpha) throws IOException {
        KnnBelief<Coordinate> beliefModel1 = new KnnBelief<Coordinate>(trainingSet, k, alpha, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
        double error1 = KnnUtils.error(crossValidation, beliefModel1) ;
        KnnBelief<Coordinate> beliefModel2 = new KnnBelief<Coordinate>(trainingSet, k, alpha + 0.01, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, Coordinate::distance) ;
        double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
        return (error2 - error1) / 0.01 ;
    }

    private List<LabelledPoint<Coordinate>> CreateList(String label, String file, int option, int option2) throws IOException {
        List<SensorValue> raw = getPoints(label, file) ;
        List<SensorValue> l1 = new ArrayList<SensorValue>() ;
        List<SensorValue> l2 = new ArrayList<SensorValue>() ;
        if (option == 1)
            l1 = toDerivative(raw) ;
        else
        if (option == 2)
            l1 = toAverage(raw) ;
        else
        if (option == 3)
            l1 = toMax(raw) ;
        else
        if (option == 4)
            l1 = toMin(raw) ;
        else
            l1 = toObv(raw) ;
        if (option2 == 1)
            l2 = toDerivative(raw) ;
        else
        if (option2 == 2)
            l2 = toAverage(raw) ;
        else
        if (option2 == 3)
            l2 = toMax(raw) ;
        else
        if (option2 == 4)
            l2 = toMin(raw) ;
        else
            l2 = toObv(raw) ;
        return fusion(l1,l2) ;
    }

    private List<LabelledPoint<Coordinate>> fusion (List<SensorValue> l1, List<SensorValue> l2) {
        if (l1.size() < l2.size())
            return fusion(l2,l1) ;
        l1 = l1.subList(l1.size() - l2.size(), l1.size() - 1) ;
        List<LabelledPoint<Coordinate>> points = new ArrayList<>(l1.size());
        for (int i = 0; i < l1.size(); i++) {
            points.add(new LabelledPoint<Coordinate> (l1.get(i).getSensor(), l1.get(i).getLabel(), l1.get(i).getTimestamp(),
                                                        new Coordinate(l1.get(i).getValue(), l2.get(i).getValue())));
        }
        return points;
    }


    /**
     * Parse the given file to extract points. One file is expected to contain only one sensor.
     *
     * @param label state in which the sample was take (i.e. presence or absence)
     * @param file  path to the file
     * @return list of points in the file
     * @throws java.io.IOException
     */
    private List<SensorValue> getPoints(String label, String file) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        InputStream resourceAsStream = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(file);
        MappingIterator<SensorValue> iterator = mapper.readValues(
                new JsonFactory().createParser(resourceAsStream),
                SensorValue.class);
        List<SensorValue> points = new ArrayList<>();
        while (iterator.hasNext()) {
            SensorValue next = iterator.next();
            next.setLabel(label);
            points.add(next);
        }
        return points;
    }

    private List<SensorValue> toObv(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size());
        for (int i = 0; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            double obv = Math.abs(point.getValue() - SENSOR_VALUE_CENTER);
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(), obv));
        }
        return points;
    }

    private List<SensorValue> toDerivative(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size() - 1);
        for (int i = 1; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            double absoluteDerivative = Math.abs(point.getValue() - prevPoint.getValue())
                    / (point.getTimestamp() - prevPoint.getTimestamp());
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(), absoluteDerivative));
        }
        return points;
    }

    private List<SensorValue> toAverage(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size() - 2);
        for (int i = 2; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            SensorValue prev2Point = sortedPoints.get(i - 2);
            double centred_average = (Math.abs(point.getValue() - SENSOR_VALUE_CENTER) + Math.abs(prevPoint.getValue() - SENSOR_VALUE_CENTER) + Math.abs(prev2Point.getValue() - SENSOR_VALUE_CENTER)) / 3;
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(), centred_average));
        }
        return points;
    }

    private double max (double a, double b, double c) {
        if (a >= b & a >= c)
            return a ;
        else if (b >= a & b >= c)
            return b ;
        else
            return c ;
    }

    private List<SensorValue> toMax(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size() - 2);
        for (int i = 2; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            SensorValue prev2Point = sortedPoints.get(i - 2);
            double maximum = max (Math.abs(point.getValue() - SENSOR_VALUE_CENTER) , Math.abs(prevPoint.getValue() - SENSOR_VALUE_CENTER) , Math.abs(prev2Point.getValue() - SENSOR_VALUE_CENTER)) ;
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(), maximum));
        }
        return points;
    }

    private double min (double a, double b, double c) {
        if (a <= b & a <= c)
            return a ;
        else if (b <= a & b <= c)
            return b ;
        else
            return c ;
    }

    private List<SensorValue> toMin(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size() - 2);
        for (int i = 2; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            SensorValue prev2Point = sortedPoints.get(i - 2);
            double minimum = min (Math.abs(point.getValue() - SENSOR_VALUE_CENTER) , Math.abs(prevPoint.getValue() - SENSOR_VALUE_CENTER) , Math.abs(prev2Point.getValue() - SENSOR_VALUE_CENTER)) ;
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(), minimum));
        }
        return points;
    }

}
