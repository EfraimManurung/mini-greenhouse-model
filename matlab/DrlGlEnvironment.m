function DrlGlEnvironment(seasonLength, firstDay, controlsFile, outdoorFile, indoorFile, fruitFile)    
    % Use for the environment in the DRL environment
    % Using createGreenLightModel
    %
    % Efraim Manurung, Information Technology Group
    % Wageningen University
    % efraim.efraimpartoginahotasi@wur.nl
    % efraim.manurung@gmail.com
    %
    % Based on:
    % David Katzin, Simon van Mourik, Frank Kempkes, and Eldert J. Van Henten. 2020. 
    % "GreenLight - An Open Source Model for Greenhouses with Supplemental Lighting: Evaluation of Heat Requirements under LED and HPS Lamps.” 
    % Biosystems Engineering 194: 61–81. https://doi.org/10.1016/j.biosystemseng.2020.03.010
    
    tic; % start the timer
    %% Set up the model

    % Choice of lamp
    lampType = 'led'; 
    
    % From IoT dataset
    [outdoor_iot, controls_iot, startTime] = loadMiniGreenhouseData(firstDay, seasonLength);

    % Load DRL controls from the .mat file
    controls = load(controlsFile);
    controls_drl = [controls.time, controls.ventilation, controls.toplights, controls.heater];

    % Ensure that the arrays are of the same length
    if size(controls_drl, 1) ~= size(controls_iot, 1)
        error('The control arrays from the .mat file do not match the expected length.');
    end
    
    % Change controls for the controls_iot from the controls_drl
    controls_iot(:,4) = controls_drl(:,2);  % Average roof ventilation aperture
    controls_iot(:,7) = controls_drl(:,3);  % Toplights on/off
    controls_iot(:,10) = controls_drl(:,4); % Boiler value / Heater
    
    % Check if the outdoofile empty or not
    if isempty(outdoorFile)
        %disp('USED OFFLINE DATASET.');
    % Try to load outdoor measurements from the .mat file
    else
            %Load outdoor measurements from the .mat file
            outdoor_file = load(outdoorFile);
            outdoor_drl = [outdoor_file.time, outdoor_file.par_out, outdoor_file.temp_out, outdoor_file.hum_out, outdoor_file.co2_out];
        
            % Function inputs:
            %   lampType        Type of lamps in the greenhouse. Choose between 
            %                   'hps', 'led', or 'none' (default is none)
            %   weather         A matrix with 8 columns, in the following format:
            %       weather(:,1)    timestamps of the input [s] in regular intervals
            %       weather(:,2)    radiation     [W m^{-2}]  outdoor global irradiation 
            %       weather(:,3)    temperature   [°C]        outdoor air temperature
            %       weather(:,4)    humidity      [kg m^{-3}] outdoor vapor concentration
            %       weather(:,5)    co2 [kg{CO2} m^{-3}{air}] outdoor CO2 concentration
            %       weather(:,6)    wind        [m s^{-1}] outdoor wind speed
            %       weather(:,7)    sky temperature [°C]
            %       weather(:,8)    temperature of external soil layer [°C]
            %       weather(:,9)    daily radiation sum [MJ m^{-2} day^{-1}]
        
            % Change for outdoor measurements
            % outdoor_iot(:,2) = outdoor_drl(:,2) * 0.0079;   % radiation     [W m^{-2}]  outdoor global irradiation source: https://www.researchgate.net/post/Howto_convert_solar_intensity_in_LUX_to_watt_per_meter_square_for_sunlight#:~:text=The%20LUX%20meter%20is%20used,of%20the%20incident%20solar%20radiation.&text=multiply%20lux%20to%200.0079%20which%20give%20you%20value%20of%20w%2Fm2.
            outdoor_iot(:,2) = outdoor_drl(:,2);
            outdoor_iot(:,3) = outdoor_drl(:,3);            % temperature   [°C]        outdoor air temperature
            outdoor_iot(:,4) = rh2vaporDens(double(outdoor_iot(:,3)), double(outdoor_drl(:,4)));  % Convert relative humidity [%] to vapor density [kg{H2O} m^{-3}]
            outdoor_iot(:,5) = co2ppm2dens(double(outdoor_iot(:,3)), double(outdoor_drl(:,5))); %co2 [kg{CO2} m^{-3}{air}] outdoor CO2 concentration

    end

    % number of seconds since beginning of year to startTime
    secsInYear = seconds(startTime-datetime(year(startTime),1,1,0,0,0));

    outdoor_iot(:,7) = outdoor_iot(:,3) - 10;
    outdoor_iot(:,8) = soilTempNl(secsInYear+outdoor_iot(:,1)); % add soil temperature

    if isempty(indoorFile)
        drl_indoor = [];
        disp('INDOOR FILE EMPTY!! FOR DEBUGING')
    else
        % Load indoor measurements from the .mat file
        indoor_file = load(indoorFile);
        drl_indoor = [indoor_file.time, indoor_file.temp_in, indoor_file.rh_in, indoor_file.co2_in];
        
        %   indoor          (optional) A 3 column matrix with:
        %       indoor(:,1)     timestamps of the input [s] in regular intervals of 300, starting with 0
        %       indoor(:,2)     temperature       [°C]             indoor air temperature
        %       indoor(:,3)     vapor pressure    [Pa]             indoor vapor concentration
        %       indoor(:,4)     co2 concentration [mg m^{-3}]      indoor vapor concentration%
        
        % DEBUG for the converted values of vapor and RH
        % Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
        % rh2_vapor = rh2vaporDens(drl_indoor(:,2), drl_indoor(:,3));
        % drl_indoor(:,3) = vaporDens2pres(drl_indoor(:,2), rh2_vapor);
        
        % convert co2 from ppm to mg m^{-3}
        % drl_indoor(:,4) = 1e6 * co2ppm2dens(drl_indoor(:,2), drl_indoor(:,4));
        
        % %Print the converted RH 
        % disp('Converted RH concentration (Pa):');
        % for i = 1:length(drl_indoor(:,3))
        %     fprintf('  %.2f\n', drl_indoor(i,3));
        % end
        % 
        % %Print the converted CO2 concentration
        % disp('Converted CO2 concentration (mg m^{-3}):');
        % for i = 1:length(drl_indoor(:,4))
        %     fprintf('  %.2f\n', drl_indoor(i,4));
        % end

    end
    
    %% Create an instance of createGreenLight with the default Vanthoor parameters
    drl_env = createGreenLightModel(lampType, outdoor_iot, startTime, controls_iot, drl_indoor);

    % Parameters for mini-greenhouse
    setParamsMiniGreenhouse(drl_env);      % set greenhouse structure
    setMiniGreenhouseLedParams(drl_env);   % set lamp params

    %% Create the the crop component of the GreenLight model 
    % Crop model
    
    % Information from setGlStates.m
    % Carbohydrates in buffer [mg{CH2O} m^{-2}]
    % addState(gl, 'cBuf');
 
    % Carbohydrates in leaves [mg{CH2O} m^{-2}]
    % addState(gl, 'cLeaf');
    
    % Carbohydrates in stem [mg{CH2O} m^{-2}]
    % addState(gl, 'cStem');
    
    % Carbohydrates in fruit [mg{CH2O} m^{-2}]
    % addState(gl, 'cFruit');
    
    % Crop development stage [°C day]
    % addState(gl, 'tCanSum');

    % Crop development stage [°C day s^{-1}]
    % Equation 8 [2]
    % setOde(gl, 'tCanSum', 1/86400*x.tCan);

    % Equation 9 [2] from A methodology for model-based greenhouse design: Part 2,
    % description and validation of a tomato yield model
    % The 24 h mean canopy temperature was approximated by a first order differential equation:
    % see the paper
    % Read also the 3.3. State variables of the model

    % Set initial values for crop
    % start with 3.12 plants/m2, assume they are each 2 g = 6240 mg/m2.
    % Check the setGlinit.m for more information
    % Default values    
    if isempty(fruitFile)
        % start with small crop
        % drl_env.x.cLeaf.val = 0.7*6240;     
        % drl_env.x.cStem.val = 0.25*6240;    
        % drl_env.x.cFruit.val = 0.05*6240;   
        % drl_env.x.tCanSum.val = 0;

        % Start with a mature crop
        drl_env.x.cLeaf.val = 2.8e5;     
        drl_env.x.cStem.val = 0.9e5;    
        drl_env.x.cFruit.val = 2.5e5;  
        drl_env.x.tCanSum.val = 3000;
    else 
        % Load DRL controls from the .mat file
        fruit_file = load(fruitFile);

        % Print the fruit growth data
        disp('Fruit growth: ');
        fprintf('          time: %.2f\n', fruit_file.time);
        fprintf('      fruit_dw: %.2f\n', fruit_file.fruit_dw);

        % Ensure that the required fields exist in fruit_file
        required_fields = {'time', 'fruit_leaf', 'fruit_stem', 'fruit_dw', 'fruit_tcansum', 'leaf_temp'};
        for i = 1:length(required_fields)
            if ~isfield(fruit_file, required_fields{i})
                error(['The fruit file does not contain the required field: ', required_fields{i}]);
            end
        end

        % Assign the loaded fruit data to the corresponding fields in drl_env
        drl_env.x.cLeaf.val = fruit_file.fruit_leaf;     
        drl_env.x.cStem.val = fruit_file.fruit_stem;    
        drl_env.x.cFruit.val = fruit_file.fruit_dw;
        drl_env.x.tCanSum.val = fruit_file.fruit_tcansum;
        drl_env.x.tCan.val = fruit_file.leaf_temp;
    end
    
    %% Run simulation
    solveFromFile(drl_env, 'ode15s');
    
    % set data to a fixed step size (5 minutes)
    drl_env = changeRes(drl_env, 300);
    
    toc;

    % Print the season length and first day
    fprintf('Season Length: %.4f day(s) \n', seasonLength);
    fprintf('Season firstDay: %.4f day(s) \n', firstDay);

    %% Extract the simulated data from the DRL environment
    time = drl_env.x.tAir.val(:, 1);                    % Time
    temp_in = drl_env.x.tAir.val(:, 2);                 % Indoor temperature
    rh_in = drl_env.a.rhIn.val(:, 2);                   % Indoor humidity
    co2_in = drl_env.a.co2InPpm.val(:, 2);              % Indoor CO2
    PAR_in = drl_env.a.rParGhSun.val(:, 2) + drl_env.a.rParGhLamp.val(:, 2); % PAR inside
    
    % For fruit growth 
    fruit_leaf = drl_env.x.cLeaf.val(:, 2);             % Fruit leaf
    fruit_stem = drl_env.x.cStem.val(:, 2);             % Fruit stem
    fruit_dw = drl_env.x.cFruit.val(:, 2);              % Fruit dry weight
    fruit_tcansum = drl_env.x.tCanSum.val(:, 2);        % Crop development stage [°C day s^{-1}]
    leaf_temp = drl_env.x.tCan.val(:, 2);              % Crop temperature [°C]
        
    % Save the extracted data to a .mat file
    save('drl-env.mat', 'time', 'temp_in', 'rh_in', 'co2_in', 'PAR_in', 'fruit_leaf', 'fruit_stem', 'fruit_dw', 'fruit_tcansum', 'leaf_temp');
    
    %% Print the values in tabular format
    fprintf('Time (s)\tIndoor Temp (°C)\tIndoor Humidity (%%)\tIndoor CO2 (ppm)\tPAR Inside (W/m²)\tFruit Dry Weight (g/m²)\tCrop Development Stage [°C day s^{-1}]\tCrop Temperature [°C]\n');
    for i = 1:length(time)
        fprintf('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n', time(i), temp_in(i), rh_in(i), co2_in(i), PAR_in(i), fruit_dw(i), fruit_tcansum(i), leaf_temp(i));
    end

    %% Clear the workspace
    % clear;
