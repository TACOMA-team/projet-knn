package fr.inria.tacoma.perf;

import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomMassFunctionGenerator {
    private final Random random;
    private final FrameOfDiscernment frame;
    private final int focalNb;

    public RandomMassFunctionGenerator(long seed, int frameSize, int focalNb) {
        random = new Random(seed);
        String[] elements = IntStream.rangeClosed(0, frameSize - 1)
                .mapToObj(suffix -> "frame" + suffix)
                .collect(Collectors.toList())
                .toArray(new String[frameSize]);
        frame = FrameOfDiscernment.newFrame("randomized", elements);
        this.focalNb = focalNb;
    }

    public List<MassFunction> generate(int functionNb) {
        List<MassFunction> functions = new ArrayList<>(functionNb);
        for (int i = 0; i < functionNb; i++) {
            functions.add(next());
        }
        return functions;
    }

    public MassFunction next() {
        MassFunction function = new MassFunctionImpl(frame);
        for (int i = 0; i < focalNb; ) {
            List<String> focalList = new ArrayList<>(frame.getStates());
            Collections.shuffle(focalList, this.random);
            focalList = focalList.subList(0, 1 + this.random.nextInt(focalList.size() - 1));
            StateSet focalElementId = this.frame.toStateSet(focalList);
            if (function.get(focalElementId) == 0.0) {
                function.addToFocal(focalElementId, random.nextDouble());

                i++;
            }
        }
        function.normalize();
        return function;
    }
}
