package fr.inria.tacoma.perf;

import fr.inria.tacoma.bft.combinations.BeliefCombination;
import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;

import java.util.ArrayList;
import java.util.List;

public class DempsterRandomCombinationPerfTest {

    private static Namespace parseArgs(String[] args) {
        ArgumentParser parser = ArgumentParsers.
                newArgumentParser("dempster-test-perf");
        parser.description("This program generates a set  of mass function pair and combine every " +
                "pair X times.");
        parser.addArgument("--pair-nb", "-p")
                .type(Integer.class)
                .setDefault(5)
                .help("number of function pair to generate (default 5)");
        parser.addArgument("--iteration-nb", "-i")
                .type(Integer.class)
                .setDefault(1000)
                .help("number of function pair to generate (default 1000)");
        parser.addArgument("--frame-size", "-s")
                .type(Integer.class)
                .setDefault(8)
                .help("size of the frame of discernment (default 8)");
        parser.addArgument("--focal-nb", "-n")
                .type(Integer.class)
                .setDefault(2)
                .help("number of focal elements in each function (default 2)");
        parser.addArgument("--seed")
                .type(Long.class)
                .setDefault(123456L)
                .help("seed for the random generator (default 123456)");

        return parser.parseArgsOrFail(args);
    }

    public static void main(String[] args) throws Exception {
        Namespace parsedArgs = parseArgs(args);

        int pairNb = parsedArgs.getInt("pair_nb");
        int iterationNb = parsedArgs.getInt("iteration_nb");
        long seed = parsedArgs.getLong("seed");
        int frameSize = parsedArgs.getInt("frame_size");
        int focalNb = parsedArgs.getInt("focal_nb");

        System.out.println("Test for dempster combination with parameters:");
        System.out.println("pair number:               " + pairNb);
        System.out.println("iteration number per pair: " + iterationNb);
        System.out.println("seed:                      " + seed);
        System.out.println("size of the frame:         " + frameSize);
        System.out.println("focal number per function: " + focalNb);
        System.out.println();


        BeliefCombination dempster = Combinations::dempster;
        RandomMassFunctionGenerator generator =
                new RandomMassFunctionGenerator(seed, frameSize, focalNb);

        List<MassFunction> functions = generatePairs(pairNb, dempster, generator);

        while (true) {
            test(iterationNb, dempster, functions);
        }
    }

    private static List<MassFunction> generatePairs(int pairNb, BeliefCombination dempster,
                                                    RandomMassFunctionGenerator generator) {
        List<MassFunction> functions = new ArrayList<>();
        while (functions.size() < pairNb * 2) {
            MassFunction function1 = generator.next();
            MassFunction function2 = generator.next();
            try {
                dempster.apply(function1, function2);
                functions.add(function1);
                functions.add(function2);
            } catch (IllegalArgumentException e) {
                //The two generated function are in full conflict, we ignore this pair
            }
        }
        return functions;
    }

    private static void test(int iterationNb, BeliefCombination dempster, List<MassFunction> functions) {

        List<Double> elapsedTimes = new ArrayList<>(functions.size());

        System.out.println("starting performance test.");


        for (int pairIndex = 0; pairIndex < functions.size() / 2; pairIndex++) {
            MassFunction function1 = functions.get(pairIndex * 2);
            MassFunction function2 = functions.get((pairIndex * 2) + 1);
            long start = System.nanoTime();
            for (int i = 0; i < iterationNb; i++) {
                dempster.apply(function1, function2);
            }
            double elapsedTime = (System.nanoTime() - start) / 1e9;
            elapsedTimes.add(elapsedTime);
        }

        double totalTime = elapsedTimes.stream().mapToDouble(Double::doubleValue).sum();
        double CombinationAvg = totalTime / (iterationNb * functions.size() / 2);

        System.out.println("total time : " + totalTime + " s");
        System.out.println("average time per combination : " + CombinationAvg * 1e6 + " µs");
        System.out.println("combinations per seconds: " + 1 / CombinationAvg);
        System.out.println("\n============================\n");
    }


}
