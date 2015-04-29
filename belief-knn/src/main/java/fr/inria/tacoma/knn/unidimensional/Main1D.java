package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.decision.CriteriaDecisionStrategy;
import fr.inria.tacoma.bft.decision.Decision;
import fr.inria.tacoma.bft.decision.DecisionStrategy;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.util.DiscountingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Main1D {

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
    public final int SENSOR_VALUE_CENTER;
    public final int MAX_X;


    public Main1D(String PRESENCE, String ABSENCE, String PRESENCE_TEST, String ABSENCE_TEST,
                      double ALPHA, int K, int FUNC,
                      double ALPHA_STEP, int K_STEP, int K_MAX,
                      double GAMMA1, double GAMMA2,
                      int MODE, int OPTION,
                      int SENSOR_VALUE_CENTER, int MAX_X) {
        this.PRESENCE = PRESENCE;
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
        this.SENSOR_VALUE_CENTER = SENSOR_VALUE_CENTER;
        this.MAX_X = MAX_X;
    }

    public void main1D() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence") ;

        // Récupère les points et calcul les features
        List<SensorValue> absence = CreateList ("absence", ABSENCE, OPTION) ;
        List<SensorValue> presence = CreateList ("presence", PRESENCE, OPTION) ;
        List<SensorValue> absence_test = CreateList ("absence", ABSENCE_TEST, OPTION) ;
        List<SensorValue> presence_test = CreateList ("presence", PRESENCE_TEST, OPTION) ;

        // Crée le jeux de test
        List<SensorValue> crossValidation = new ArrayList<>() ;
        crossValidation.addAll(absence_test) ;
        crossValidation.addAll(presence_test) ;

        // Crée le jeux d'apprentissage
        List<SensorValue> trainingSet = new ArrayList<>() ;
        trainingSet.addAll(absence) ;
        trainingSet.addAll(presence) ;

        // Sélectionne la bonne action selon le mode
        if (MODE == 1)
            showTheOne(frame, trainingSet, crossValidation) ;
        else if (MODE == 2)
            varAlphaK(frame, trainingSet, crossValidation) ;
        else if (MODE == 3)
            bestAlphaK(frame, trainingSet, crossValidation) ;
        else if (MODE == 4)
            grad(frame, trainingSet, crossValidation, GAMMA1, GAMMA2) ;
    }

    //Affiche la fonction de masse.
    private void showTheOne(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation) {
        KnnBelief<Double> beliefModel = new KnnBelief<Double>(trainingSet, K, ALPHA, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
        double error = KnnUtils.error(crossValidation, beliefModel) ;
        System.out.println("k = " + K) ;
        System.out.println("alpha = " + ALPHA) ;
        System.out.println("error = " + error) ;
        show(beliefModel) ;
    }

    // Ecrit dans le fichier "Result.txt" l'erreur en fonction de k et alpha
    // Affiche sur la sortie standard les meilleures valeurs de k et alpha et lerreur associée
    private void varAlphaK(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation) throws IOException {
        int k_mem = 0 ;
        double alpha_mem = 0. ;
        double error_mem = Double.POSITIVE_INFINITY ;
        File file = new File("Result.txt") ;
        file.createNewFile() ;
        FileWriter writer = new FileWriter(file) ;
                writer.write("k,alpha,error\n") ;
        for (int k = 1 ; k < trainingSet.size() & k <= K_MAX ; k += K_STEP) {
            for (double alpha = ALPHA_STEP ; alpha < 1. ; alpha += ALPHA_STEP) {
                KnnBelief<Double> beliefModel = new KnnBelief<Double>(trainingSet, k, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
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

    // Affiche sur la sortie standard les meilleures valeurs de k et alpha et l'erreur associée
    private void bestAlphaK(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation) throws IOException {
        int k_mem = 0 ;
        double alpha_mem = 0. ;
        double error_mem = Double.POSITIVE_INFINITY ;
        for (int k = 1 ; k < trainingSet.size() & k <= K_MAX ; k += K_STEP) {
            for (double alpha = ALPHA_STEP ; alpha < 1. ; alpha += ALPHA_STEP) {
                KnnBelief<Double> beliefModel = new KnnBelief<Double>(trainingSet, k, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
                double error = KnnUtils.error(crossValidation, beliefModel) ;
                if (error < error_mem) {
                    k_mem = k ;
                    alpha_mem = alpha ;
                    error_mem = error ;
                }
            }
        }
        System.out.println("Dimension = 1 ; Option = " + OPTION + " ; Fontion = " + FUNC + " -> k = " + k_mem + " ; alpha = " + alpha_mem + " ; error = " + error_mem) ;
    }

    // Applique la methode du gradient
    private void grad(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation,
                             double gamma1, double gamma2) throws IOException {
        int k_mem = 1 ;
        double alpha_mem = 0.01 ;
        double error_mem = KnnUtils.error(crossValidation, new KnnBelief<Double>(trainingSet, k_mem, alpha_mem, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b))) ;
        boolean stop = false ;
        int count = 0;
        while (! stop) {
//        System.out.println("begin " + count) ;
            int k0 = k_mem ;
            double alpha0 = alpha_mem ;
            double dk = derivate_k(frame, trainingSet, crossValidation, k_mem, alpha_mem, error_mem) ;
//       System.out.println("dk = " + dk) ;
            double dalpha = derivate_alpha(frame, trainingSet, crossValidation, k_mem, alpha_mem, error_mem) ;
//       System.out.println("dalpha = " + dalpha) ;
            int k1 = Math.max(k_mem - ((int) (Math.floor(dk * gamma1))), 1);
            int k2 = Math.max(k_mem - ((int) (Math.ceil(dk * gamma1))), 1);
            double alpha = Math.max(Math.min(alpha_mem - dalpha * gamma2, 0.98), 0.01) ;
            KnnBelief<Double> beliefModel1 = new KnnBelief<Double>(trainingSet, k1, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
            double error1 = KnnUtils.error(crossValidation, beliefModel1) ;
//       System.out.println("error1 = " + error1) ;
            KnnBelief<Double> beliefModel2 = new KnnBelief<Double>(trainingSet, k2, alpha, FUNC, frame,
                            KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
            double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
//       System.out.println("error2 = " + error2) ;
            if(error1 < error_mem){
                k_mem = k1 ;
                alpha_mem = alpha ;
                error_mem = error1 ;
            }
            if(error2 < error_mem){
                k_mem = k2 ;
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

    private double derivate_k(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation,
                             int k, double alpha, double error0) throws IOException {
        KnnBelief<Double> beliefModel2 = new KnnBelief<Double>(trainingSet, k+1, alpha, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
        double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
        return error2 - error0 ;
    }

    private double derivate_alpha(FrameOfDiscernment frame, List<SensorValue> trainingSet, List<SensorValue> crossValidation,
                             int k, double alpha, double error0) throws IOException {
        KnnBelief<Double> beliefModel2 = new KnnBelief<Double>(trainingSet, k, alpha + 0.01, FUNC, frame,
                        KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b)) ;
        double error2 = KnnUtils.error(crossValidation, beliefModel2) ;
        return (error2 - error0) / 0.01 ;
    }

    // Affiche la fonction de masse donnée en argument
    private void show(SensorBeliefModel<Double> model) {
        ChartPanel chartPanel = JfreeChartDisplay1D.getChartPanel(model, 512, 0, MAX_X);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }


    // Récupère la liste de points depuis le fichier d'entrée et calcule les features
    private List<SensorValue> CreateList(String label, String file, int option) throws IOException {
        List<SensorValue> raw = getPoints(label, file) ;
        if (option == 1)
            raw = toDerivative(raw) ;
        else
        if (option == 2)
            raw = toAverage(raw) ;
        else
        if (option == 3)
            raw = toMax(raw) ;
        else
        if (option == 4)
            raw = toMin(raw) ;
        else
            raw.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER))) ;
        return raw ;
    }

    // Extrait la liste des point du fichiers d'entrée
    private List<SensorValue> getPoints(String label, String file) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, true);
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

    // Calcule la liste des dérivées
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

    // Calcule la liste des moyennes des trois dernières valeurs
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

    // Calcule la liste des maximums des trois dernières valeurs
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

    // Calcule la liste des minimums des trois dernières valeurs
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
