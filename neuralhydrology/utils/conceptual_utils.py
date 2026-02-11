import torch

def daily_jensen_haise_pet(T_avg: torch.Tensor, S_rad: torch.Tensor) -> torch.Tensor:
    """Calculate potential evapotranspiration (PET) using the Jensen-Haise Method from Jensen et al. (2016).
        
        This method estimates PET from shortwave radiation and mean temperature. It is fully differentiable
        and meant to be implemented in the training loop of a conceptual model. Taken from jensen_haise method
        from PyEt python package. 
        
        Parameters
        ----------
        T_avg : torch.Tensor
            Average temperature (°C)
        S_rad : torch.Tensor
            Shortwave radiation (W/m²)
        
        Outputs
        -------
        torch.Tensor
            Estimated potential evapotranspiration (PET) in mm/day.
            
        References
        ----------
        M. E. Jensen and R. G. Allen, Evaporation, evapotranspiration, and irrigation water requirements. 
        American Society of Civil Engineers (ASCE), Jan. 2016, pp. 632–633,isbn: 9780784479209. 
        doi: 10.1061/9780784414057.
        
        M. Vremec, M. Collenteur, and S. Birk, Technical note: Improved handling of potential evapotranspiration 
        in hydrological studies with PyEt. (2023). Python. doi: https://doi.org/10.5194/hess-2022-417.

    """
    SECONDS_to_DAYS = 24*3600
    J_to_MJ = 1/1000000
    shortRad_MJ_per_m2_day = (
        S_rad * SECONDS_to_DAYS * J_to_MJ
    ) # convert shortwave radiation [W/m^2] to [MJ/m^2 day]
    lambd_latent_heat_vaporization_MJ_per_kg = 2.501 - 0.002361 * T_avg # linear imperical relationship
    pet_mm_per_day = (0.025 * shortRad_MJ_per_m2_day * (T_avg - (-3.0)) / lambd_latent_heat_vaporization_MJ_per_kg)
    pet_mm_per_day_mask = pet_mm_per_day < 0  # make mask for negative PET
    pet_mm_per_day = torch.where(pet_mm_per_day_mask, 0, pet_mm_per_day)  # clip negative PET to 0
        
    return pet_mm_per_day