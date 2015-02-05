package fr.inria.tacoma.bft.combinations;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static fr.inria.tacoma.bft.combinations.Combinations.duboisAndPrade;
import static org.junit.Assert.*;

public class DuboisPradeCombinationTest {

    private FrameOfDiscernment frame;
    private MassFunction mass1;
    private MassFunction mass2;
    private MassFunction mass3;

    @Before
    public void setup() {
        frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");

        mass1 = new MassFunctionImpl(frame);
        mass2 = new MassFunctionImpl(frame);
        mass3 = new MassFunctionImpl(frame);

        mass1.addToFocal(frame.toStateSet("layingDown"), 0.3);
        mass1.addToFocal(frame.toStateSet("sitting", "layingDown"), 0.5);
        mass1.putRemainingOnIgnorance();

        mass2.addToFocal(frame.toStateSet("sitting"), 0.6);
        mass2.putRemainingOnIgnorance();

        mass3.addToFocal(frame.toStateSet("standing"), 0.3);
        mass3.putRemainingOnIgnorance();
    }

    @Test
    public void DuboisPradeCombination_Combine2Functions_ReturnsExpectedResult() {
        MassFunction expected = new MassFunctionImpl(frame);

        expected.addToFocal(frame.toStateSet("layingDown"), 0.12);
        expected.addToFocal(frame.toStateSet("sitting"), 0.42);
        expected.addToFocal(frame.toStateSet("sitting", "layingDown"), 0.38);
        expected.addToFocal(frame.fullIgnoranceSet(), 0.08);

        assertEquals(expected, duboisAndPrade(mass1, mass2));
    }

    @Test
    public void DuboisPradeCombinationCombine2Function_ReturnExpectedResult2() {
        MassFunction expected = new MassFunctionImpl(frame);
        expected.addToFocal(frame.toStateSet("sitting", "standing"), 0.18);
        expected.addToFocal(frame.toStateSet("sitting"), 0.42);
        expected.addToFocal(frame.toStateSet("standing"), 0.12);
        expected.addToFocal(frame.fullIgnoranceSet(), 0.28);

        assertEquals(expected, duboisAndPrade(mass2, mass3));
    }

    @Test
    public void combineWithList_with2MassFunction_GivesSameResultAsUsualFunction() {
        List<MassFunction> massFunctions = Arrays.asList(mass1,mass2);
        assertEquals(duboisAndPrade(massFunctions), duboisAndPrade(mass1, mass2));
    }


    @Test
    public void combineWithList_moreThan2Function_DoesNotDependOnOrder() {
        List<MassFunction> massFunctions = Arrays.asList(mass1,mass2,mass3);
        List<MassFunction> massFunctions2 = Arrays.asList(mass1,mass3,mass2);

        assertEquals(duboisAndPrade(massFunctions), duboisAndPrade(massFunctions2));
    }
}
