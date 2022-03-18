classdef IndependentProductDitribution < IndependentProductDistribution
   
    properties
    end
    
    methods 
        function IP = IndependentProductDitribution(factors)
            warning('IndependentProductDitribution is misspelled. Correct name!')
            IP@IndependentProductDistribution(factors);
        end %endFunction
    end
end